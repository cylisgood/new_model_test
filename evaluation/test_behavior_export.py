"""
行為區段輸出（behavior_export.py：dry-run 與 Supabase 上傳）的單元測試。
Supabase 用本機假伺服器（127.0.0.1，隨機 port）代替，不會連到真的 Supabase、不需要攝影機或 GPU；
只有其中一個測試會載入 main_model（要幾秒），用真正的 update_behavior_segments 產生區段，
確認欄位格式跟輸出端對得上。

用法（在專案根目錄）：
  venv\\Scripts\\python.exe evaluation\\test_behavior_export.py
"""

import contextlib
import io
import json
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import behavior_export as be  # noqa: E402

PET_UUID = "123e4567-e89b-12d3-a456-426614174000"
OWNER_EMAIL = "owner@example.com"
OWNER_PASSWORD = "very-secret-pw"
ANON_KEY = "anon-key-xyz"


def fixed_clock():
    return datetime(2026, 11, 2, 9, 30, 0, tzinfo=timezone.utc)


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def seg(action, start, end):
    return {"action": action, "start_frame": start, "end_frame": end}


def wait_until(predicate, timeout=5.0, what="條件成立"):
    start = time.time()

    while time.time() - start < timeout:
        if predicate():
            return

        time.sleep(0.02)

    raise AssertionError(f"等了 {timeout} 秒仍然沒有：{what}")


# ---- 環境變數 ----

def test_export_mode():
    assert be.read_export_mode({}) == "off"
    assert be.read_export_mode({"BEHAVIOR_EXPORT": ""}) == "off"
    assert be.read_export_mode({"BEHAVIOR_EXPORT": "off"}) == "off"
    assert be.read_export_mode({"BEHAVIOR_EXPORT": " DryRun "}) == "dryrun"
    assert be.read_export_mode({"BEHAVIOR_EXPORT": "Supabase"}) == "supabase"

    try:
        be.read_export_mode({"BEHAVIOR_EXPORT": "supabse"})
    except ValueError as e:
        assert "supabse" in str(e)
    else:
        raise AssertionError("打錯字要報錯，不能默默當成關閉")


def test_pet_id():
    assert be.read_pet_id({}) is None
    assert be.read_pet_id({"PET_ID": "  "}) is None
    assert be.read_pet_id({"PET_ID": PET_UUID.upper()}) == PET_UUID  # 統一成小寫標準格式

    try:
        be.read_pet_id({"PET_ID": "not-a-uuid"})
    except ValueError:
        pass
    else:
        raise AssertionError("PET_ID 不是 uuid 要報錯（資料表的 pet_id 欄位是 uuid）")


def test_min_seconds_setting():
    assert be.read_min_seconds({}) == 0.5
    assert be.read_min_seconds({"EXPORT_MIN_SECONDS": "0"}) == 0.0
    assert be.read_min_seconds({"EXPORT_MIN_SECONDS": "1.5"}) == 1.5

    for bad in ("-1", "abc"):
        try:
            be.read_min_seconds({"EXPORT_MIN_SECONDS": bad})
        except ValueError:
            pass
        else:
            raise AssertionError(f"EXPORT_MIN_SECONDS={bad!r} 要報錯")


def test_supabase_settings_validation():
    good = {
        "SUPABASE_URL": "https://abc.supabase.co/",
        "SUPABASE_ANON_KEY": ANON_KEY,
        "SUPABASE_EMAIL": OWNER_EMAIL,
        "SUPABASE_PASSWORD": OWNER_PASSWORD,
        "PET_ID": PET_UUID,
    }
    settings = be.read_supabase_settings(good)
    assert settings.url == "https://abc.supabase.co" and settings.pet_id == PET_UUID

    # 後台 Data API 頁面複製到的網址帶 /rest/v1，貼成那樣也要能用
    pasted = be.read_supabase_settings({**good, "SUPABASE_URL": "https://abc.supabase.co/rest/v1/"})
    assert pasted.url == "https://abc.supabase.co"

    # 帳密、金鑰不能被 repr / print 洩漏
    text = repr(settings)
    assert OWNER_PASSWORD not in text and OWNER_EMAIL not in text and ANON_KEY not in text

    # 缺什麼就報什麼，而且只列變數名稱，不印其他設定的值
    try:
        be.read_supabase_settings({k: v for k, v in good.items() if k not in ("SUPABASE_PASSWORD", "PET_ID")})
    except ValueError as e:
        message = str(e)
        assert "SUPABASE_PASSWORD" in message and "PET_ID" in message
        assert OWNER_EMAIL not in message and ANON_KEY not in message
    else:
        raise AssertionError("缺設定要報錯")

    # 範本的 <...> 沒拿掉（實際發生過：金鑰前後多了尖括號，結果被閘道拒絕）要直接講出來
    try:
        be.read_supabase_settings({**good, "SUPABASE_ANON_KEY": f"<{ANON_KEY}>"})
    except ValueError as e:
        assert "SUPABASE_ANON_KEY" in str(e) and "尖括號" in str(e)
        assert ANON_KEY not in str(e)
    else:
        raise AssertionError("保留 <...> 要報錯")

    # 只查寵物清單時不需要 PET_ID
    assert be.read_supabase_settings({k: v for k, v in good.items() if k != "PET_ID"}, need_pet_id=False).pet_id is None

    # 帳密不能走明文：只有 localhost 可以用 http
    for bad_url in ("http://abc.supabase.co", "ftp://abc.supabase.co", "not a url"):
        try:
            be.read_supabase_settings({**good, "SUPABASE_URL": bad_url})
        except ValueError:
            pass
        else:
            raise AssertionError(f"SUPABASE_URL={bad_url!r} 要報錯")

    assert be.read_supabase_settings({**good, "SUPABASE_URL": "http://localhost:54321"}).url == "http://localhost:54321"


# ---- 本機輸出內容 ----

def test_rows_match_table_columns():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.jsonl"
        exporter = be.DryRunExporter(path, PET_UUID, clock=fixed_clock)

        n = exporter.export_new({1: [seg("walking", 30, 90)]}, fps=15.0)
        assert n == 1

        (row,) = read_rows(path)
        assert str(uuid.UUID(row["id"])) == row["id"]                       # 用戶端產生的 uuid
        assert row["pet_id"] == PET_UUID
        assert row["start_seconds"] == 2.0 and row["end_seconds"] == 6.0   # 幀數 / fps
        assert row["behavior"] == "walking"                                 # 模型原始標籤
        assert row["created_at"] == "2026-11-02T09:30:00+00:00"
        assert row["_track_id"] == 1


def test_only_new_closed_segments_are_exported():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.jsonl"
        exporter = be.DryRunExporter(path, clock=fixed_clock)
        segments = {1: [seg("sitting", 0, 60)]}

        assert exporter.export_new(segments, fps=15.0) == 1
        assert exporter.export_new(segments, fps=15.0) == 0       # 沒有新區段，不重複輸出

        segments[1].append(seg("lying", 61, 200))
        segments[2] = [seg("standing", 10, 100)]                  # 新出現的另一隻
        assert exporter.export_new(segments, fps=15.0) == 2

        rows = read_rows(path)
        assert [r["behavior"] for r in rows] == ["sitting", "lying", "standing"]
        assert rows[0]["pet_id"] is None                           # 沒設 PET_ID 就是 null
        assert len({r["id"] for r in rows}) == 3                   # 每筆 id 不同


def test_non_behavior_and_zero_length_are_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.jsonl"
        exporter = be.DryRunExporter(path, clock=fixed_clock)

        n = exporter.export_new(
            {1: [seg("unknown", 0, 30), seg("analyzing", 31, 40), seg("standing", 50, 50), seg("running", 60, 120)]},
            fps=15.0,
        )

        assert n == 1 and exporter.skipped == 3
        assert [r["behavior"] for r in read_rows(path)] == ["running"]


def test_min_seconds_filter_drops_flicker():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.jsonl"
        exporter = be.DryRunExporter(path, clock=fixed_clock, min_seconds=0.5)

        # 一幀的「站」（約 0.07 秒）是剛追蹤到時的第一個標籤一閃而過，不輸出；後面真正的行為照常輸出
        n = exporter.export_new({1: [seg("standing", 135, 136), seg("lying", 137, 275)]}, fps=15.0)

        assert n == 1 and exporter.skipped == 1
        assert [r["behavior"] for r in read_rows(path)] == ["lying"]


def test_write_failure_does_not_raise_and_retries_later():
    with tempfile.TemporaryDirectory() as tmp:
        bad_path = Path(tmp)                                       # 是資料夾，開檔會失敗
        exporter = be.DryRunExporter(bad_path, clock=fixed_clock)
        segments = {1: [seg("walking", 0, 30)]}

        assert exporter.export_new(segments, fps=15.0) == 0        # 失敗只印警告，不丟例外

        exporter.path = Path(tmp) / "out.jsonl"                    # 路徑恢復後，同一批會補寫
        assert exporter.export_new(segments, fps=15.0) == 1
        assert len(read_rows(exporter.path)) == 1


def test_bad_fps_exports_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        exporter = be.DryRunExporter(Path(tmp) / "out.jsonl", clock=fixed_clock)
        assert exporter.export_new({1: [seg("walking", 0, 30)]}, fps=0) == 0
        assert exporter.export_new({1: [seg("walking", 0, 30)]}, fps=None) == 0


def test_build_exporter_off_returns_none():
    assert be.build_exporter("off", "unused.jsonl") is None


# ---- Supabase：用本機假伺服器 ----

class StubSupabase:
    """本機假的 Supabase：只實作登入、新增紀錄、查寵物，並記錄收到的每個請求。"""

    def __init__(self):
        self.requests = []          # (方法, 路徑, 標頭, 內容)
        self.insert_statuses = []   # 依序回給「新增紀錄」的狀態碼，用完就回 201
        self.login_status = 200
        self.expires_in = 3600
        self.pets = [{"id": PET_UUID, "name": "麻糬"}]
        self.token_count = 0
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _read_json(self):
                length = int(self.headers.get("Content-Length") or 0)
                return json.loads(self.rfile.read(length)) if length else None

            def _send(self, status, payload=None):
                data = b"" if payload is None else json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                body = self._read_json()
                stub.requests.append(("POST", self.path, dict(self.headers), body))

                if self.path.startswith("/auth/v1/token"):
                    if "grant_type=password" in self.path and stub.login_status != 200:
                        return self._send(stub.login_status, {"error": "invalid_grant"})

                    stub.token_count += 1
                    return self._send(200, {
                        "access_token": f"tok{stub.token_count}",
                        "refresh_token": f"ref{stub.token_count}",
                        "expires_in": stub.expires_in,
                    })

                if self.path.startswith("/rest/v1/pet_behavior_logs"):
                    status = stub.insert_statuses.pop(0) if stub.insert_statuses else 201
                    return self._send(status, None if status < 400 else {"message": "stub error"})

                self._send(404, {})

            def do_GET(self):
                stub.requests.append(("GET", self.path, dict(self.headers), None))

                if self.path.startswith("/rest/v1/pets"):
                    pets = stub.pets

                    if "id=eq." in self.path:
                        wanted = self.path.split("id=eq.")[1].split("&")[0]
                        pets = [p for p in pets if p["id"] == wanted]

                    return self._send(200, pets)

                self._send(404, {})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()

    def inserts(self):
        return [r for r in self.requests if r[0] == "POST" and r[1].startswith("/rest/v1/pet_behavior_logs")]

    def logins(self):
        return [r for r in self.requests if r[0] == "POST" and "grant_type=password" in r[1]]

    def refreshes(self):
        return [r for r in self.requests if r[0] == "POST" and "grant_type=refresh_token" in r[1]]


def header(request, name):
    for key, value in request[2].items():
        if key.lower() == name.lower():
            return value

    return None


def sample_rows(count=2):
    return [
        {
            "id": str(uuid.uuid4()),
            "pet_id": PET_UUID,
            "start_seconds": float(i * 10),
            "end_seconds": float(i * 10 + 3),
            "behavior": "walking",
            "created_at": "2026-11-02T09:30:00+00:00",
            "_track_id": 1,
            "_session_started_at": "2026-11-02T09:25:00+00:00",
        }
        for i in range(count)
    ]


@contextlib.contextmanager
def rig(configure=None, backoff=(0.01,), expires_in=None):
    """假 Supabase + 用戶端 + 上傳器；離開時一定關掉執行緒和伺服器。"""
    stub = StubSupabase()

    if expires_in is not None:
        stub.expires_in = expires_in

    if configure:
        configure(stub)

    settings = be.SupabaseSettings(
        url=stub.url, anon_key=ANON_KEY, email=OWNER_EMAIL, password=OWNER_PASSWORD, pet_id=PET_UUID,
    )
    client = be.SupabaseClient(settings, opener=urllib.request.build_opener(urllib.request.ProxyHandler({})))
    logs = []
    uploader = be.SupabaseUploader(client, backoff=backoff, log=logs.append)

    try:
        yield stub, uploader, logs
    finally:
        uploader.close(timeout=3)
        stub.stop()


def test_login_and_insert_request_shape():
    with rig() as (stub, uploader, logs):
        rows = sample_rows(2)
        assert uploader.submit(rows) is True
        wait_until(lambda: uploader.uploaded == 2, what="兩筆上傳完成")

        (login,) = stub.logins()
        assert login[3] == {"email": OWNER_EMAIL, "password": OWNER_PASSWORD}
        assert header(login, "apikey") == ANON_KEY

        (insert,) = stub.inserts()
        assert "on_conflict=id" in insert[1]                                  # 重送不會重複寫入
        assert header(insert, "apikey") == ANON_KEY
        assert header(insert, "Authorization") == "Bearer tok1"
        assert header(insert, "Prefer") == "resolution=ignore-duplicates,return=minimal"

        # 只送資料表有的欄位，底線開頭的檢查用欄位不能上傳
        assert [set(item) for item in insert[3]] == [set(be.TABLE_COLUMNS)] * 2
        assert [item["id"] for item in insert[3]] == [r["id"] for r in rows]

        # 啟動時就確認過寵物，而且帳密、金鑰沒有出現在任何提示訊息裡
        assert any("麻糬" in line for line in logs)
        text = "\n".join(logs)
        assert OWNER_PASSWORD not in text and ANON_KEY not in text


def test_retries_server_errors_with_same_ids():
    with rig(lambda s: s.insert_statuses.extend([500, 503])) as (stub, uploader, logs):
        rows = sample_rows(2)
        uploader.submit(rows)
        wait_until(lambda: uploader.uploaded == 2, what="重試後上傳成功")

        inserts = stub.inserts()
        assert len(inserts) == 3                                              # 失敗兩次 + 成功一次
        assert inserts[0][3] == inserts[1][3] == inserts[2][3]                # 每次送的 id 都一樣
        assert uploader.dropped == 0


def test_expired_token_triggers_new_token():
    with rig(lambda s: s.insert_statuses.append(401)) as (stub, uploader, logs):
        uploader.submit(sample_rows(1))
        wait_until(lambda: uploader.uploaded == 1, what="換權杖後上傳成功")

        first, second = stub.inserts()
        assert header(first, "Authorization") != header(second, "Authorization")
        assert uploader.dropped == 0


def test_rejected_batch_is_dropped_but_next_batch_goes_through():
    with rig(lambda s: s.insert_statuses.append(403)) as (stub, uploader, logs):
        uploader.submit(sample_rows(2))                                       # 被擋（例如權限不對）
        uploader.submit(sample_rows(1))
        wait_until(lambda: uploader.pending() == 0, what="兩批都處理完")

        assert len(stub.inserts()) == 2                                       # 被擋的那批沒有重試
        assert uploader.dropped == 2 and uploader.uploaded == 1
        assert any("被拒絕" in line for line in logs)


def test_bad_credentials_disable_uploads():
    with rig(lambda s: setattr(s, "login_status", 400)) as (stub, uploader, logs):
        wait_until(lambda: uploader.disabled, what="登入失敗後停用上傳")

        assert uploader.submit(sample_rows(1)) is False
        assert stub.inserts() == []                                           # 不會拿壞帳密一直去打
        assert len(stub.logins()) == 1
        text = "\n".join(logs)
        assert "登入失敗" in text and OWNER_PASSWORD not in text


def test_rejected_key_is_reported_as_key_problem():
    # 401 / 403 是閘道擋的（金鑰有問題），要跟「帳密錯誤（400）」講得不一樣，才不會讓人去改密碼
    with rig(lambda s: setattr(s, "login_status", 401)) as (stub, uploader, logs):
        wait_until(lambda: uploader.disabled, what="金鑰被拒後停用上傳")

        text = "\n".join(logs)
        assert "SUPABASE_ANON_KEY" in text and "SUPABASE_PASSWORD" not in text
        assert ANON_KEY not in text and OWNER_PASSWORD not in text


def test_wrong_pet_id_disables_uploads():
    with rig(lambda s: setattr(s, "pets", [])) as (stub, uploader, logs):
        wait_until(lambda: uploader.disabled, what="PET_ID 對不到寵物後停用上傳")

        assert any("PET_ID" in line for line in logs)
        assert uploader.submit(sample_rows(1)) is False


def test_short_lived_token_is_refreshed():
    with rig(expires_in=30) as (stub, uploader, logs):                        # 比提早換新的 60 秒還短，每次都視為過期
        uploader.submit(sample_rows(1))
        wait_until(lambda: uploader.uploaded == 1, what="用換新後的權杖上傳成功")

        assert len(stub.refreshes()) >= 1
        assert len(stub.logins()) == 1                                        # 換新靠 refresh token，不用重新輸入密碼


def test_close_does_not_hang_while_retrying():
    with rig(lambda s: s.insert_statuses.extend([500] * 100), backoff=(30,)) as (stub, uploader, logs):
        uploader.submit(sample_rows(1))
        wait_until(lambda: len(stub.inserts()) >= 1, what="第一次嘗試")

        started = time.time()
        uploader.close(timeout=5)

        assert time.time() - started < 3                                      # 正在等 30 秒重試也要馬上收工
        assert uploader.dropped == 1


def test_supabase_exporter_keeps_local_copy_and_uploads():
    with rig() as (stub, uploader, logs), tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.jsonl"
        exporter = be.SupabaseExporter(path, uploader, PET_UUID, clock=fixed_clock)

        assert exporter.export_new({1: [seg("lying", 0, 60), seg("walking", 61, 120)]}, fps=15.0) == 2
        wait_until(lambda: uploader.uploaded == 2, what="兩筆上傳完成")

        local = read_rows(path)
        (insert,) = stub.inserts()
        assert [r["id"] for r in local] == [item["id"] for item in insert[3]]  # 本機留底跟上傳的是同一批
        assert all("_track_id" not in item for item in insert[3])
        assert all(item["pet_id"] == PET_UUID for item in insert[3])


def test_list_pets_command():
    with rig() as (stub, uploader, logs):
        env = {
            "SUPABASE_URL": stub.url, "SUPABASE_ANON_KEY": ANON_KEY,
            "SUPABASE_EMAIL": OWNER_EMAIL, "SUPABASE_PASSWORD": OWNER_PASSWORD,
        }

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = be.main(["list-pets"], env=env)

        assert code == 0 and PET_UUID in out.getvalue() and "麻糬" in out.getvalue()
        assert OWNER_PASSWORD not in out.getvalue()

        stub.login_status = 400
        with contextlib.redirect_stdout(io.StringIO()):
            assert be.main(["list-pets"], env=env) == 1                       # 登入失敗
            assert be.main(["oops"], env=env) == 2                            # 用法錯誤
            assert be.main(["list-pets"], env={}) == 2                        # 沒設環境變數


# ---- 跟模型真正產生的區段格式對得上 ----

def test_works_with_real_segment_builder():
    import main_model as mm  # noqa: E402  (會載入 deeplabcut，要幾秒)

    behavior_segments, active_segments = {}, {}

    # 站 0~29 幀、坐 30~59 幀、最後躺到 89 幀（最後這段還在進行中）
    for frame_id in range(90):
        action = "standing" if frame_id < 30 else "sitting" if frame_id < 60 else "lying"
        mm.update_behavior_segments(behavior_segments, active_segments, 1, action, frame_id)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.jsonl"
        exporter = be.DryRunExporter(path, clock=fixed_clock)

        assert exporter.export_new(behavior_segments, fps=30.0) == 2   # 進行中的 lying 還不輸出
        assert [r["behavior"] for r in read_rows(path)] == ["standing", "sitting"]

        # 收尾：write_behavior_log 會把進行中的區段收進 behavior_segments，再輸出一次就補上了
        mm.write_behavior_log(behavior_segments, active_segments, 30.0, "test", Path(tmp) / "log.txt")
        assert exporter.export_new(behavior_segments, fps=30.0) == 1

        rows = read_rows(path)
        assert [r["behavior"] for r in rows] == ["standing", "sitting", "lying"]
        assert rows[2]["start_seconds"] == 2.0 and rows[2]["end_seconds"] == 2.97


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"通過：{t.__name__}")
    print(f"\n{len(tests)} 個測試全部通過")
