"""
把 .env 設定檔讀進環境變數，不需要多裝套件。

main_model.py 載入時會呼叫 load_env_files()，所以 python main_model.py、uvicorn server:app、
python behavior_export.py list-pets 啟動時設定都已經在了，不用每次開終端機都重設環境變數。

優先順序（前面的優先；每個檔案只補上「還沒設定」的變數）：
  1. 真正的環境變數       永遠優先，檔案蓋不過它
  2. ENV_FILE 指定的檔案
  3. ~/.petbehavior/.env   在 OneDrive 以外，密碼建議放這裡
  4. 專案資料夾的 .env     已被 .gitignore 排除；但專案若在 OneDrive 底下，檔案會被同步到雲端

檔案格式：一行一個「名稱=值」，# 開頭是註解，可以有 export 前綴，值前後成對的引號會拿掉；
不支援行尾註解（密碼裡可能有 #）。值是空的行視為沒設定，所以範本只填名稱也沒關係。
檔案要存成 UTF-8。

只會印出讀到哪個檔案、設定了哪些變數的「名稱」，不會印出值。
SKIP_ENV_FILE=1 可以整個不讀設定檔（測試用）。
"""

import os
import re
from pathlib import Path

ENV_FILE_VAR = "ENV_FILE"
SKIP_VAR = "SKIP_ENV_FILE"
SECRET_VARS = ("SUPABASE_PASSWORD",)
NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _log(message):
    """純提示訊息：主控台編碼印不出 emoji 時退而求其次，不能因為印不出來就讓程式啟動失敗。"""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", "replace").decode("ascii"))


def parse_env_text(text):
    """
    回傳 (values, bad_lines)：values 是 {名稱: 值}（同名以後面的為準），
    bad_lines 是格式不對的行號。只回報行號、不回報內容，避免把密碼印出來。
    """
    values = {}
    bad_lines = []

    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()

        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export "):].lstrip()

        name, separator, value = line.partition("=")
        name = name.strip()

        if not separator or not NAME_PATTERN.match(name):
            bad_lines.append(number)
            continue

        value = value.strip()

        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]

        values[name] = value

    return values, bad_lines


def default_env_paths(environ=None, project_dir=None, home=None):
    """設定檔的搜尋順序，前面的優先。"""
    environ = os.environ if environ is None else environ
    project_dir = Path(__file__).resolve().parent if project_dir is None else Path(project_dir)
    paths = []

    explicit = environ.get(ENV_FILE_VAR, "").strip()

    if explicit:
        paths.append(Path(explicit))

    try:
        home = Path.home() if home is None else Path(home)
        paths.append(home / ".petbehavior" / ".env")
    except RuntimeError:   # 找不到使用者資料夾就只剩專案資料夾那份
        pass

    paths.append(project_dir / ".env")

    return paths


def _in_onedrive(path):
    return any(part.lower().startswith("onedrive") for part in Path(path).resolve().parts)


def load_env_files(paths=None, environ=None, log=_log, project_dir=None, home=None):
    """
    依序讀設定檔，把還沒設定的變數放進 environ（預設是 os.environ）。
    回傳這次讀到的 [(檔案路徑, 設定了哪些變數名稱)]。project_dir、home 只有測試會傳。
    """
    environ = os.environ if environ is None else environ

    if environ.get(SKIP_VAR, "").strip() == "1":
        return []

    if paths is None:
        explicit = environ.get(ENV_FILE_VAR, "").strip()

        if explicit and not Path(explicit).is_file():
            log(f"⚠️ ENV_FILE 指定的設定檔不存在：{explicit}")

        paths = default_env_paths(environ, project_dir=project_dir, home=home)

    loaded = []

    for path in paths:
        path = Path(path)

        if not path.is_file():
            continue

        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            log(f"⚠️ 設定檔 {path} 不是 UTF-8 編碼，已略過（用 PowerShell 的 > 建立的檔案常是 UTF-16，請改存成 UTF-8）")
            continue
        except OSError as e:
            log(f"⚠️ 讀不到設定檔 {path}（{type(e).__name__}），已略過")
            continue

        values, bad_lines = parse_env_text(text)
        applied = []

        for name, value in values.items():
            # 空值視為沒設定；真正的環境變數有值就不蓋掉
            if not value or environ.get(name, "").strip():
                continue

            environ[name] = value
            applied.append(name)

        loaded.append((path, applied))
        log(f"📄 已載入設定檔：{path}（{'、'.join(applied) if applied else '沒有新增的變數'}）")

        if bad_lines:
            log(f"⚠️ {path} 第 {'、'.join(str(n) for n in bad_lines)} 行格式不對（要「名稱=值」），已略過")

        if any(name in applied for name in SECRET_VARS) and _in_onedrive(path):
            log(f"⚠️ {path} 含密碼，但在 OneDrive 資料夾裡，會被同步到雲端；建議改放 ~/.petbehavior/.env")

    return loaded
