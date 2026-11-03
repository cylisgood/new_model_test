"""
設定檔載入（env_file.py）的單元測試：解析格式、優先順序、不蓋過真正的環境變數、不外洩值、
OneDrive 警告，以及確認 main_model.py 的載入順序、behavior_export.py list-pets 真的吃得到設定檔。
不連網路、不需要攝影機或 GPU，也不載入 DeepLabCut。

用法（在專案根目錄）：
  venv\\Scripts\\python.exe evaluation\\test_env_file.py
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
import env_file as ef  # noqa: E402

SECRET = "pw-with=equals#and-hash"


def write(path, text, encoding="utf-8"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode(encoding))
    return path


def quiet(_message):
    pass


# ---- 解析 ----

def test_parse_basic():
    values, bad_lines = ef.parse_env_text(
        "# 註解\n"
        "\n"
        "BEHAVIOR_EXPORT=supabase\n"
        "export PET_ID = 123e4567\n"
        'SUPABASE_EMAIL="owner@example.com"\n'
        "SUPABASE_NOTE='keeps trailing space '\n"
        f"SUPABASE_PASSWORD={SECRET}\n"
        "EMPTY=\n"
        "DUP=first\n"
        "DUP=second\n"
    )

    assert bad_lines == []
    assert values == {
        "BEHAVIOR_EXPORT": "supabase",
        "PET_ID": "123e4567",
        "SUPABASE_EMAIL": "owner@example.com",
        "SUPABASE_NOTE": "keeps trailing space ",
        "SUPABASE_PASSWORD": SECRET,    # 值裡的 = 和 # 都保留
        "EMPTY": "",
        "DUP": "second",                # 同名以後面的為準
    }


def test_parse_bad_lines_report_numbers_only():
    values, bad_lines = ef.parse_env_text("GOOD=1\nthis is not valid\n=novalue\n1BAD=x\nOK=2\n")

    assert values == {"GOOD": "1", "OK": "2"}
    assert bad_lines == [2, 3, 4]


def test_bom_and_crlf():
    with tempfile.TemporaryDirectory() as tmp:
        path = write(Path(tmp) / ".env", "﻿A=1\r\nB=2\r\n")
        environ = {}

        ef.load_env_files([path], environ, log=quiet)

        assert environ == {"A": "1", "B": "2"}


# ---- 載入與優先順序 ----

def test_real_env_wins_over_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = write(Path(tmp) / ".env", "PET_ID=from-file\nOTHER=x\n")
        environ = {"PET_ID": "from-shell"}

        loaded = ef.load_env_files([path], environ, log=quiet)

        assert environ == {"PET_ID": "from-shell", "OTHER": "x"}
        assert loaded == [(path, ["OTHER"])]


def test_empty_real_env_does_not_block_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = write(Path(tmp) / ".env", "PET_ID=from-file\n")
        environ = {"PET_ID": "  "}

        ef.load_env_files([path], environ, log=quiet)

        assert environ["PET_ID"] == "from-file"


def test_earlier_file_wins_and_empty_values_dont_block():
    with tempfile.TemporaryDirectory() as tmp:
        first = write(Path(tmp) / "first.env", "A=first\nB=\n")
        second = write(Path(tmp) / "second.env", "A=second\nB=from-second\n")
        environ = {}

        loaded = ef.load_env_files([first, second], environ, log=quiet)

        assert environ == {"A": "first", "B": "from-second"}
        assert loaded == [(first, ["A"]), (second, ["B"])]


def test_missing_files_and_directories_are_ignored():
    with tempfile.TemporaryDirectory() as tmp:
        logs = []

        loaded = ef.load_env_files([Path(tmp) / "nope.env", Path(tmp)], {}, log=logs.append)

        assert loaded == []
        assert logs == []


def test_skip_env_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = write(Path(tmp) / ".env", "A=1\n")
        environ = {"SKIP_ENV_FILE": "1"}

        assert ef.load_env_files([path], environ, log=quiet) == []
        assert "A" not in environ


def test_non_utf8_file_is_skipped_with_hint():
    with tempfile.TemporaryDirectory() as tmp:
        path = write(Path(tmp) / ".env", "A=1\n", encoding="utf-16")   # PowerShell 的 > 常產生這種
        logs, environ = [], {}

        assert ef.load_env_files([path], environ, log=logs.append) == []
        assert environ == {}
        assert any("UTF-8" in message for message in logs)


# ---- 預設路徑 ----

def test_default_paths_order():
    assert ef.default_env_paths({"ENV_FILE": "X.env"}, project_dir="P", home="H") == [
        Path("X.env"),
        Path("H") / ".petbehavior" / ".env",
        Path("P") / ".env",
    ]
    assert ef.default_env_paths({}, project_dir="P", home="H") == [
        Path("H") / ".petbehavior" / ".env",
        Path("P") / ".env",
    ]


def test_default_search_reads_home_before_project_and_warns_on_missing_explicit():
    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp) / "home"
        project = Path(tmp) / "project"
        write(home / ".petbehavior" / ".env", "A=from-home\n")
        write(project / ".env", "A=from-project\nB=from-project\n")
        logs, environ = [], {"ENV_FILE": str(Path(tmp) / "typo.env")}

        ef.load_env_files(environ=environ, log=logs.append, project_dir=project, home=home)

        assert environ["A"] == "from-home"
        assert environ["B"] == "from-project"
        assert any("ENV_FILE" in message and "不存在" in message for message in logs)


# ---- 不外洩 ----

def test_logs_never_contain_values():
    with tempfile.TemporaryDirectory() as tmp:
        path = write(
            Path(tmp) / ".env",
            f"SUPABASE_PASSWORD={SECRET}\nPET_ID=123e4567-e89b\nmy secret garbage line\n",
        )
        logs = []

        ef.load_env_files([path], {}, log=logs.append)

        text = "\n".join(logs)
        assert SECRET not in text
        assert "123e4567-e89b" not in text
        assert "my secret garbage" not in text
        assert "SUPABASE_PASSWORD" in text      # 名稱可以印，方便確認有沒有讀到
        assert "第 3 行" in text


def test_warns_when_password_file_is_in_onedrive():
    with tempfile.TemporaryDirectory() as tmp:
        with_password = write(Path(tmp) / "OneDrive" / "Desktop" / "proj" / ".env", f"SUPABASE_PASSWORD={SECRET}\n")
        without_password = write(Path(tmp) / "OneDrive" / "other" / ".env", "PET_ID=1\n")
        outside = write(Path(tmp) / "plain" / ".env", f"SUPABASE_PASSWORD={SECRET}\n")

        def warned(path):
            logs = []
            ef.load_env_files([path], {}, log=logs.append)
            return any("同步到雲端" in message for message in logs)

        assert warned(with_password)
        assert not warned(without_password)     # 沒有密碼就不用警告
        assert not warned(outside)


# ---- 跟其他程式接得起來 ----

def test_main_model_loads_env_before_reading_it():
    source = (REPO_DIR / "main_model.py").read_text(encoding="utf-8")
    call = source.index("env_file.load_env_files()")
    first_reads = [i for i in (source.find("os.getenv("), source.find("os.environ"), source.find("behavior_export.read_")) if i >= 0]

    assert first_reads and call < min(first_reads), "載入設定檔必須在 main_model.py 第一次讀環境變數之前"


def test_list_pets_cli_reads_the_env_file():
    with tempfile.TemporaryDirectory() as tmp:
        config = write(
            Path(tmp) / "config.env",
            "SUPABASE_URL=http://127.0.0.1:9\n"          # 沒人在聽的 port：只要能連到這裡就代表設定檔有被讀到
            "SUPABASE_ANON_KEY=anon-xyz\n"
            "SUPABASE_EMAIL=owner@example.com\n"
            f"SUPABASE_PASSWORD={SECRET}\n",
        )
        env = {k: v for k, v in os.environ.items() if not k.startswith("SUPABASE_") and k not in ("ENV_FILE", "SKIP_ENV_FILE")}
        env["ENV_FILE"] = str(config)
        env["USERPROFILE"] = env["HOME"] = tmp                   # 不要讀到這台電腦上真正的 ~/.petbehavior/.env
        env["PYTHONIOENCODING"] = "utf-8"

        result = subprocess.run(
            [sys.executable, str(REPO_DIR / "behavior_export.py"), "list-pets"],
            cwd=tmp, env=env, capture_output=True, text=True, encoding="utf-8", timeout=60,
        )
        output = result.stdout + result.stderr

        assert result.returncode == 1, output
        assert "連不上 Supabase" in output, output          # 有讀到設定、嘗試連線才會是這個訊息
        assert "需要設定環境變數" not in output, output
        assert SECRET not in output, output


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"通過：{t.__name__}")
    print(f"\n{len(tests)} 個測試全部通過")
