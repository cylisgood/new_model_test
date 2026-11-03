"""
把行為區段整理成 yourpetapp 的 Supabase 資料表 pet_behavior_logs 的列格式，並輸出。

兩種模式（環境變數 BEHAVIOR_EXPORT，預設關閉）：
  dryrun    只把「剛結束的行為區段」附加寫到本機 jsonl，不連 Supabase
  supabase  同樣寫一份本機 jsonl（留底），並另外上傳到 Supabase 的 pet_behavior_logs

欄位對應（App 端 supabase/migrations/20261004000000_pet_behavior_logs.sql）：
  id              用戶端產生的 uuid；重試時同一筆 id 不變，搭配 on_conflict=id 就不會重複寫入
  pet_id          寵物 uuid（環境變數 PET_ID）
  start_seconds   區段開始秒數 = start_frame / fps（跟行為紀錄.txt 同一套算法）
  end_seconds     區段結束秒數 = end_frame / fps
  behavior        模型原始標籤，不是 /current 對應後的標籤（App 的顏色表用原始標籤）
  created_at      寫出時間
開頭是底線的欄位（_track_id、_session_started_at）只是本機檢查用，不會上傳。

Supabase 模式的身分：用寵物擁有者的 App 帳號（email + 密碼）登入，權限受 RLS 限制，
只能新增自己寵物的紀錄。所有金鑰、帳密都從環境變數讀，不寫進程式、不印出來。
"""

import json
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# App 的 lib/petBehaviorLogs.js BEHAVIOR_META 認得的標籤；
# unknown、analyzing 是「資料不足」的暫時標籤，不是真的行為，不輸出
EXPORT_BEHAVIORS = ("lying", "walking", "running", "standing", "sitting")

TABLE = "pet_behavior_logs"
TABLE_COLUMNS = ("id", "pet_id", "start_seconds", "end_seconds", "behavior", "created_at")
SUPABASE_ENV_NAMES = ("SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_EMAIL", "SUPABASE_PASSWORD")
DEFAULT_MIN_SECONDS = 0.5   # 比這短的區段多半是剛追蹤到時第一個標籤一閃而過，不輸出


def _log(message):
    """純提示訊息：主控台編碼不支援（例如 cp950 印不出 emoji）時退而求其次，不能因為印不出來就中斷分析。"""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", "replace").decode("ascii"))


# ============================================================
# 環境變數
# ============================================================

def read_export_mode(env=None):
    """
    BEHAVIOR_EXPORT：沒設或 off 代表關閉，dryrun 只寫本機檔，supabase 另外上傳。
    其他值直接報錯，避免打錯字卻以為有在輸出。
    """
    env = os.environ if env is None else env
    raw = env.get("BEHAVIOR_EXPORT", "").strip().lower()

    if raw in ("", "off"):
        return "off"

    if raw in ("dryrun", "supabase"):
        return raw

    raise ValueError(f"BEHAVIOR_EXPORT 只支援 off、dryrun、supabase，收到：{raw!r}")


def read_pet_id(env=None):
    """PET_ID：這支攝影機對應的寵物 uuid。沒設回傳 None；格式不對直接報錯。"""
    env = os.environ if env is None else env
    raw = env.get("PET_ID", "").strip()

    if not raw:
        return None

    return str(uuid.UUID(raw))


def read_min_seconds(env=None):
    """EXPORT_MIN_SECONDS：區段至少多長才輸出，預設 0.5 秒；0 代表不過濾。"""
    env = os.environ if env is None else env
    raw = env.get("EXPORT_MIN_SECONDS", "").strip()

    if not raw:
        return DEFAULT_MIN_SECONDS

    value = float(raw)

    if value < 0:
        raise ValueError("EXPORT_MIN_SECONDS 不能是負數")

    return value


@dataclass(frozen=True)
class SupabaseSettings:
    url: str
    anon_key: str = field(repr=False)
    email: str = field(repr=False)
    password: str = field(repr=False)
    pet_id: Optional[str] = None


def read_supabase_settings(env=None, need_pet_id=True):
    """
    BEHAVIOR_EXPORT=supabase 需要的設定。缺哪個就報錯（只列變數名稱，不印值）。
    SUPABASE_URL 一定要 https（只有 localhost 可以用 http），避免帳密以明文送出。
    """
    env = os.environ if env is None else env
    values = {name: env.get(name, "").strip() for name in SUPABASE_ENV_NAMES}
    missing = [name for name, value in values.items() if not value]
    pet_id = read_pet_id(env)

    if need_pet_id and pet_id is None:
        missing.append("PET_ID")

    if missing:
        raise ValueError("Supabase 模式需要設定環境變數：" + "、".join(missing))

    # 範本裡的 <...> 是要整個換掉的，連尖括號一起留著會被當成真的值（金鑰多半會因此被拒絕）
    placeholders = [name for name, value in values.items() if value.startswith("<") and value.endswith(">")]

    if placeholders:
        raise ValueError("這些環境變數還是範本的 <...> 寫法，請換成真的值並拿掉尖括號：" + "、".join(placeholders))

    url = values["SUPABASE_URL"].rstrip("/")

    # 後台 Data API 頁面複製到的是帶 /rest/v1 的網址；程式自己會加路徑，這裡拿掉
    if url.endswith("/rest/v1"):
        url = url[: -len("/rest/v1")]

    parsed = urllib.parse.urlparse(url)

    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("SUPABASE_URL 格式不對，應該像 https://xxxx.supabase.co")

    if parsed.scheme != "https" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("SUPABASE_URL 必須是 https（只有 localhost 可以用 http）")

    return SupabaseSettings(
        url=url,
        anon_key=values["SUPABASE_ANON_KEY"],
        email=values["SUPABASE_EMAIL"],
        password=values["SUPABASE_PASSWORD"],
        pet_id=pet_id,
    )


# ============================================================
# 本機輸出（dry-run，也是 Supabase 模式的留底）
# ============================================================

class DryRunExporter:
    """
    每分析完一段影片呼叫一次 export_new()，只輸出「上次之後新結束」的區段。
    behavior_segments[追蹤編號] 只會往後附加已結束的區段，所以用「已輸出幾個」當游標就夠了；
    還在進行中的區段（active_segments）不輸出，等它結束或程式收尾時才會進到 behavior_segments。
    """

    def __init__(self, path, pet_id=None, clock=None, min_seconds=0.0, resolver=None):
        self.path = Path(path)
        self.pet_id = pet_id
        self.min_seconds = min_seconds
        # resolver(track_id, segment) -> pet_identity.Decision；一支攝影機有多隻寵物時，
        # 由它決定這段行為屬於哪隻寵物，沒給就全部記在 pet_id（舊行為）
        self.resolver = resolver
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.session_started_at = self._now()
        self._exported = {}   # 追蹤編號 -> 已處理過幾個區段
        self.skipped = 0      # 不是真行為（unknown 等）或太短而略過的區段數
        self.unidentified = 0  # 認不出是哪隻寵物、因此不上傳的區段數（本機 jsonl 仍有留底）

    def _now(self):
        return self._clock().isoformat(timespec="seconds")

    def _to_row(self, track_id, segment, fps):
        if segment["action"] not in EXPORT_BEHAVIORS:
            return None

        start_seconds = segment["start_frame"] / fps
        end_seconds = segment["end_frame"] / fps

        # App 算的是 end - start，長度 0 的區段加了也不影響統計
        if end_seconds <= start_seconds or end_seconds - start_seconds < self.min_seconds:
            return None

        pet_id = self.pet_id
        identity = None

        if self.resolver is not None:
            identity = self.resolver(track_id, segment)
            pet_id = identity.pet_id

        row = {
            "id": str(uuid.uuid4()),
            "pet_id": pet_id,
            "start_seconds": round(start_seconds, 2),
            "end_seconds": round(end_seconds, 2),
            "behavior": segment["action"],
            "created_at": self._now(),
            "_track_id": track_id,
            "_session_started_at": self.session_started_at,
        }

        if identity is not None:
            row["_identity"] = {"reason": identity.reason, "scores": identity.scores, "samples": identity.samples}

        return row

    def _append_file(self, rows):
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as e:
            _log(f"⚠️ 行為區段寫檔失敗（不影響分析）：{e}")
            return False

        return True

    def _deliver(self, rows):
        """把這批區段送出去；回傳 False 代表這次沒成功，游標不前進，下次連同這批一起重試。"""
        if not self._append_file(rows):
            return False

        _log(f"📝 [dry-run] 新增 {len(rows)} 筆行為區段 → {self.path}")
        return True

    def export_new(self, behavior_segments, fps):
        """回傳這次輸出幾筆。輸出失敗只印警告，不能影響行為分析本身。"""
        if not fps or fps <= 0:
            return 0

        rows = []
        cursors = {}
        skipped = 0

        for track_id, segments in behavior_segments.items():
            for segment in segments[self._exported.get(track_id, 0):]:
                row = self._to_row(track_id, segment, fps)

                if row is None:
                    skipped += 1
                else:
                    rows.append(row)

            cursors[track_id] = len(segments)

        if rows and not self._deliver(rows):
            return 0

        self._exported.update(cursors)
        self.skipped += skipped

        if self.resolver is not None:
            self.unidentified += sum(1 for row in rows if row["pet_id"] is None)

        return len(rows)

    def close(self, timeout=None):
        pass


# ============================================================
# Supabase 用戶端（只用標準函式庫，不需要多裝套件）
# ============================================================

class SupabaseError(Exception):
    def __init__(self, status, body=""):
        super().__init__(f"HTTP {status}: {str(body)[:200]}")
        self.status = status
        self.body = body


class SupabaseAuthError(SupabaseError):
    """登入被拒絕（帳密錯誤等）；重試也不會好，不要一直打登入端點。"""


def describe_auth_error(error):
    """401 / 403 是閘道擋下來的，多半是金鑰有問題；帳密錯誤則是 400。"""
    if error.status in (401, 403):
        return f"登入失敗（HTTP {error.status}）：金鑰被拒絕，多半是 SUPABASE_ANON_KEY 沒貼完整、或前後多了符號"

    return f"登入失敗（HTTP {error.status}）：email 或密碼不對，請檢查 SUPABASE_EMAIL / SUPABASE_PASSWORD"


class SupabaseClient:
    def __init__(self, settings, opener=None, clock=None, timeout=15):
        self.settings = settings
        self._opener = opener or urllib.request.build_opener()
        self._clock = clock or time.time
        self._timeout = timeout
        self._access_token = None
        self._refresh_token = None
        self._expires_at = 0.0

    def _call(self, method, path, body=None, headers=None):
        data = None if body is None else json.dumps(body).encode("utf-8")
        all_headers = {"apikey": self.settings.anon_key, "Content-Type": "application/json"}
        all_headers.update(headers or {})
        request = urllib.request.Request(self.settings.url + path, data=data, method=method, headers=all_headers)

        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                return response.status, response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            raise SupabaseError(e.code, e.read().decode("utf-8", "replace")) from None

    def _store_session(self, raw):
        data = json.loads(raw)
        self._access_token = data["access_token"]
        self._refresh_token = data.get("refresh_token")
        # 提早 60 秒換新，避免剛好在上傳途中過期
        self._expires_at = self._clock() + max(0, int(data.get("expires_in", 3600)) - 60)

    def sign_in(self):
        try:
            _, raw = self._call(
                "POST",
                "/auth/v1/token?grant_type=password",
                {"email": self.settings.email, "password": self.settings.password},
            )
        except SupabaseError as e:
            if e.status in (400, 401, 403, 422):
                raise SupabaseAuthError(e.status, e.body) from None
            raise

        self._store_session(raw)

    def invalidate_token(self):
        self._access_token = None
        self._expires_at = 0.0

    def _auth_headers(self):
        if not self._access_token or self._clock() >= self._expires_at:
            refreshed = False

            if self._refresh_token:
                try:
                    _, raw = self._call(
                        "POST",
                        "/auth/v1/token?grant_type=refresh_token",
                        {"refresh_token": self._refresh_token},
                    )
                    self._store_session(raw)
                    refreshed = True
                except SupabaseError:
                    pass  # refresh token 失效就退回重新登入

            if not refreshed:
                self.sign_in()

        return {"Authorization": f"Bearer {self._access_token}"}

    def insert_logs(self, rows):
        """on_conflict=id + ignore-duplicates：同一批重送（例如回應掉了）不會寫出重複的列。"""
        payload = [{key: row[key] for key in TABLE_COLUMNS if key in row} for row in rows]
        headers = dict(self._auth_headers())
        headers["Prefer"] = "resolution=ignore-duplicates,return=minimal"
        self._call("POST", f"/rest/v1/{TABLE}?on_conflict=id", payload, headers)

    def find_pet(self, pet_id):
        query = urllib.parse.urlencode({"select": "id,name", "id": f"eq.{pet_id}"})
        _, raw = self._call("GET", f"/rest/v1/pets?{query}", None, self._auth_headers())
        return json.loads(raw)

    def list_pets(self):
        _, raw = self._call("GET", "/rest/v1/pets?select=id,name&order=created_at.asc", None, self._auth_headers())
        return json.loads(raw)

    def list_pets_with_avatars(self):
        """寵物辨認用：多拿種類和大頭照網址。"""
        query = "/rest/v1/pets?select=id,name,species,avatar_url&order=created_at.asc"
        _, raw = self._call("GET", query, None, self._auth_headers())
        return json.loads(raw)


class SupabaseUploader:
    """
    背景執行緒上傳，分析執行緒只負責把區段丟進佇列，Supabase 慢或斷線都不會拖慢分析。
    網路錯誤、429、5xx 會一直重試（間隔逐漸拉長）；其他 4xx（例如權限被拒）重試也不會好，直接丟掉並印出原因；
    登入被拒絕（帳密錯）就整個停用，不再打登入端點。丟掉的區段在本機 jsonl 留底還找得到。
    """

    def __init__(self, client, max_queue=200, backoff=(2, 5, 15, 30, 60), log=_log):
        self.client = client
        self._log = log
        self._backoff = backoff
        self._queue = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self.uploaded = 0
        self.dropped = 0
        self.disabled = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def pending(self):
        return self._queue.unfinished_tasks

    def submit(self, rows):
        if self.disabled:
            self.dropped += len(rows)
            return False

        try:
            self._queue.put_nowait(rows)
        except queue.Full:
            self.dropped += len(rows)
            self._log("⚠️ [supabase] 上傳佇列滿了，這批只保留在本機檔")
            return False

        return True

    def close(self, timeout=10.0):
        self._stop.set()  # 重試中的批次不要再等了

        try:
            self._queue.put(None, timeout=1)
        except queue.Full:
            return

        self._thread.join(timeout)

    def _disable(self, reason):
        self.disabled = True
        self._log(f"⚠️ [supabase] 已停用上傳：{reason}（本機檔仍會繼續留底）")

    def _startup_check(self):
        """啟動時先登入、確認 PET_ID 真的是這個帳號的寵物，有問題馬上說，不用等到第一筆區段。"""
        settings = self.client.settings

        try:
            self.client.sign_in()

            if settings.pet_id:
                pets = self.client.find_pet(settings.pet_id)

                if not pets:
                    self._disable("PET_ID 不是這個帳號底下的寵物，寫入一定會被拒絕")
                    return

                self._log(f"☁️ [supabase] 已登入，寫入目標寵物：{pets[0].get('name')}")
                return

            self._log("☁️ [supabase] 已登入")
        except SupabaseAuthError as e:
            self._disable(describe_auth_error(e))
        except (SupabaseError, OSError, ValueError, KeyError) as e:
            self._log(f"⚠️ [supabase] 啟動時沒連上，之後會自動重試：{e}")

    def _run(self):
        self._startup_check()

        while True:
            rows = self._queue.get()

            try:
                if rows is None:
                    return

                if self.disabled:
                    self.dropped += len(rows)
                else:
                    self._upload(rows)
            finally:
                self._queue.task_done()

    def _wait_to_retry(self, attempt):
        delay = self._backoff[min(attempt, len(self._backoff) - 1)]
        return not self._stop.wait(delay)   # False 代表正在關閉，不用再等

    def _upload(self, rows):
        attempt = 0
        reauth_tries = 0

        while True:
            try:
                self.client.insert_logs(rows)
                self.uploaded += len(rows)
                self._log(f"☁️ [supabase] 已寫入 {len(rows)} 筆行為區段")
                return
            except SupabaseAuthError as e:
                self.dropped += len(rows)
                self._disable(describe_auth_error(e))
                return
            except SupabaseError as e:
                if e.status == 401 and reauth_tries < 3:
                    reauth_tries += 1
                    self.client.invalidate_token()   # 權杖失效，下一輪重新換新
                    continue

                if e.status in (401, 408, 429) or e.status >= 500:
                    self._log(f"⚠️ [supabase] 暫時寫不進去，稍後重試：{e}")
                else:
                    self.dropped += len(rows)
                    self._log(f"⚠️ [supabase] 這批 {len(rows)} 筆被拒絕，不再重試（本機檔有留底）：{e}")
                    return
            except OSError as e:
                self._log(f"⚠️ [supabase] 連線失敗，稍後重試：{e}")
            except Exception as e:  # 預期外的錯誤也不能讓上傳執行緒死掉
                self.dropped += len(rows)
                self._log(f"⚠️ [supabase] 上傳時發生預期外的錯誤，這批不再重試：{e!r}")
                return

            if not self._wait_to_retry(attempt):
                self.dropped += len(rows)
                self._log("⚠️ [supabase] 程式關閉中，這批只保留在本機檔")
                return

            attempt += 1


class SupabaseExporter(DryRunExporter):
    """本機 jsonl 照寫（留底），同時把同一批丟給背景上傳。"""

    def __init__(self, path, uploader, pet_id, clock=None, min_seconds=0.0, resolver=None):
        super().__init__(path, pet_id, clock=clock, min_seconds=min_seconds, resolver=resolver)
        self.uploader = uploader

    def _deliver(self, rows):
        self._append_file(rows)   # 留底寫不成也不擋上傳

        # 認不出是哪隻寵物的不上傳（pet_id 是空的資料庫也不收），只留在本機 jsonl
        uploadable = [row for row in rows if row["pet_id"] is not None]

        if len(uploadable) < len(rows):
            _log(f"🐾 {len(rows) - len(uploadable)} 段行為認不出是哪隻寵物，不上傳（本機留底）")

        if not uploadable:
            return True

        try:
            self.uploader.submit(uploadable)
        except Exception as e:
            _log(f"⚠️ [supabase] 交給上傳執行緒失敗（不影響分析）：{e!r}")

        return True

    def close(self, timeout=10.0):
        self.uploader.close(timeout)


def build_exporter(mode, path, pet_id=None, supabase=None, min_seconds=0.0, resolver=None):
    """依 BEHAVIOR_EXPORT 建立輸出器；off 回傳 None。resolver 見 DryRunExporter。"""
    if mode == "dryrun":
        _log(f"📝 [dry-run] 行為區段會輸出到 {path}（不連 Supabase）")
        return DryRunExporter(path, pet_id, min_seconds=min_seconds, resolver=resolver)

    if mode == "supabase":
        uploader = SupabaseUploader(SupabaseClient(supabase))
        _log(f"☁️ [supabase] 行為區段會寫入 {TABLE}，並在本機保留一份 {path}")
        return SupabaseExporter(path, uploader, supabase.pet_id, min_seconds=min_seconds, resolver=resolver)

    return None


# ============================================================
# 指令列工具：python behavior_export.py list-pets
# 用 SUPABASE_* 環境變數登入，印出這個帳號底下的寵物 id 和名字，方便填 PET_ID
# ============================================================

def main(argv=None, env=None):
    argv = sys.argv[1:] if argv is None else argv

    if argv != ["list-pets"]:
        _log("用法：python behavior_export.py list-pets")
        return 2

    try:
        client = SupabaseClient(read_supabase_settings(env, need_pet_id=False))
        client.sign_in()
        pets = client.list_pets()
    except ValueError as e:
        _log(f"設定有問題：{e}")
        return 2
    except SupabaseAuthError as e:
        _log(describe_auth_error(e))
        return 1
    except (SupabaseError, OSError) as e:
        _log(f"連不上 Supabase：{e}")
        return 1

    if not pets:
        _log("這個帳號底下沒有寵物")

    for pet in pets:
        _log(f"{pet.get('id')}  {pet.get('name')}")

    return 0


if __name__ == "__main__":
    import env_file

    env_file.load_env_files()   # 讓 list-pets 也吃得到 .env，不用另外設環境變數
    sys.exit(main())
