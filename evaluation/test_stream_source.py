"""
即時串流來源（HUB_AI_STREAM / RTSP_URL、實測幀率、重連）的單元測試。
全部用假的環境變數、假的時鐘和假的攝影機，不需要連到真的 Hub，也不需要 DeepLabCut 推論。

用法（在專案根目錄）：
  venv\\Scripts\\python.exe evaluation\\test_stream_source.py
"""

import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main_model as mm  # noqa: E402  (會載入 deeplabcut，要幾秒)


# ---- 環境變數 ----

def test_stream_url_precedence():
    assert mm.read_stream_url({"HUB_AI_STREAM": "rtsp://a/x", "RTSP_URL": "rtsp://b/y"}) == "rtsp://a/x"
    assert mm.read_stream_url({"RTSP_URL": "rtsp://b/y"}) == "rtsp://b/y"
    assert mm.read_stream_url({"HUB_AI_STREAM": "rtsp://a/x"}) == "rtsp://a/x"
    assert mm.read_stream_url({"HUB_AI_STREAM": "  ", "RTSP_URL": "rtsp://b/y"}) == "rtsp://b/y"  # 空白視為沒設
    assert mm.read_stream_url({}) == ""


def test_source_mode_defaults():
    # 沒指定 SOURCE_MODE：有網址就用串流（不要默默打開本機鏡頭），沒網址才用 webcam
    assert mm.read_source_mode({}, stream_url="rtsp://a/x") == "stream"
    assert mm.read_source_mode({}, stream_url="") == "webcam"
    assert mm.read_source_mode({"SOURCE_MODE": ""}, stream_url="rtsp://a/x") == "stream"

    # 明確指定的優先
    assert mm.read_source_mode({"SOURCE_MODE": "webcam"}, stream_url="rtsp://a/x") == "webcam"
    assert mm.read_source_mode({"SOURCE_MODE": "VIDEO"}, stream_url="") == "video"

    # 打錯的值維持原本的行為（退回 stream），之後會因為沒網址而明確報錯
    assert mm.read_source_mode({"SOURCE_MODE": "typo"}, stream_url="") == "stream"


def test_redact_url():
    assert mm.redact_url("rtsp://user:secret@100.1.2.3:8554/cam") == "rtsp://100.1.2.3:8554/cam"
    assert mm.redact_url("rtsp://100.1.2.3:8554/cam") == "rtsp://100.1.2.3:8554/cam"
    assert "secret" not in mm.redact_url("http://u:secret@host/video")


def _with_module_values(**values):
    old = {k: getattr(mm, k) for k in values}
    for k, v in values.items():
        setattr(mm, k, v)
    return old


def test_resolve_stream_source_hides_credentials_in_label():
    old = _with_module_values(SOURCE_MODE="stream", STREAM_URL="rtsp://user:secret@100.1.2.3:8554/cam_ai")
    try:
        mode, source, label = mm.resolve_input_source()
        assert mode == "stream" and source.startswith("rtsp://user:secret@")  # 連線要用完整網址
        assert "secret" not in label and "user" not in label                  # 但標籤（檔名 / 日誌）不能含帳密
    finally:
        _with_module_values(**old)


def test_missing_url_error_names_both_env_vars():
    old = _with_module_values(SOURCE_MODE="stream", STREAM_URL="")
    try:
        try:
            mm.resolve_input_source()
        except ValueError as e:
            assert "HUB_AI_STREAM" in str(e) and "RTSP_URL" in str(e), str(e)
        else:
            raise AssertionError("沒設網址時應該要報錯")
    finally:
        _with_module_values(**old)


# ---- 實測幀率 ----

class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeCap:
    """每次 read() 讓假時鐘前進一個間隔，模擬串流到手的節奏。"""

    def __init__(self, clock, intervals):
        self.clock = clock
        self.intervals = list(intervals)

    def read(self):
        if not self.intervals:
            return False, None
        self.clock.t += self.intervals.pop(0)
        return True, "frame"


def test_measure_fps_ignores_initial_burst():
    # 剛連上先連續灌進 20 幀（50 fps 的緩衝），之後穩定 12 fps：量出來要是 12，不是被緩衝拉高
    clock = FakeClock()
    cap = FakeCap(clock, [0.02] * 20 + [1 / 12] * 200)
    fps, last = mm.measure_capture_fps(cap, seconds=5.0, settle=1.0, clock=clock)
    assert fps is not None and 11.9 < fps < 12.1, fps
    assert last == "frame"


def test_measure_fps_not_the_declared_value():
    clock = FakeClock()
    cap = FakeCap(clock, [1 / 16.6] * 200)
    fps, _ = mm.measure_capture_fps(cap, seconds=5.0, settle=1.0, clock=clock)
    assert 16.4 < fps < 16.8, fps


def test_measure_fps_too_few_frames_returns_none():
    clock = FakeClock()
    cap = FakeCap(clock, [0.1, 0.1])  # 只讀到 2 幀就斷線
    fps, last = mm.measure_capture_fps(cap, seconds=5.0, settle=1.0, clock=clock)
    assert fps is None and last == "frame"


# ---- open_capture ----

def test_open_capture_requests_tcp():
    # 連到一個沒人聽的 port：很快就失敗，只是要檢查它設定的選項
    cap = mm.open_capture("rtsp://127.0.0.1:9/none", "stream")
    try:
        assert not cap.isOpened()
    finally:
        cap.release()

    options = os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
    assert options.startswith("rtsp_transport;tcp"), options
    assert "udp" not in options, options


# ---- 重連 ----

def test_reconnect_retries_until_success():
    calls = {"n": 0}

    class Cap:
        def __init__(self, ok):
            self.ok = ok
            self.released = False

        def isOpened(self):
            return self.ok

        def read(self):
            return (True, "frame") if self.ok else (False, None)

        def release(self):
            self.released = True

    made = []

    def fake_open(source, mode):
        calls["n"] += 1
        cap = Cap(ok=calls["n"] >= 3)  # 前兩次連不上，第三次成功
        made.append(cap)
        return cap

    original = mm.open_capture
    mm.open_capture = fake_open
    try:
        cap, frame = mm.reconnect_capture("src", "stream", threading.Event(), wait=0.01)
    finally:
        mm.open_capture = original

    assert calls["n"] == 3 and cap is made[2] and frame == "frame"
    assert made[0].released and made[1].released and not made[2].released  # 失敗的要釋放，成功的留著用


def test_reconnect_stops_when_asked():
    stop = threading.Event()
    stop.set()

    original = mm.open_capture
    mm.open_capture = lambda *a: (_ for _ in ()).throw(AssertionError("被要求停止後不該再嘗試連線"))
    try:
        assert mm.reconnect_capture("src", "stream", stop, wait=0.01) == (None, None)
    finally:
        mm.open_capture = original


# ---- /current 的狀態要反映來源是否連著 ----

def test_current_status_follows_source_connection():
    from collections import deque

    tracks = {1: {"action_history": deque(["sitting"] * 5), "current_action": "sitting"}}

    mm.set_source_connected(False)
    mm.update_latest_result(tracks)
    assert mm.latest_result["status"] == "offline", mm.latest_result  # 斷線期間分析舊片段，不能變回 online

    mm.set_source_connected(True)
    mm.update_latest_result(tracks)
    assert mm.latest_result["status"] == "online", mm.latest_result

    mm.mark_offline()
    assert mm.source_connected is False and mm.latest_result["status"] == "offline"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"通過：{t.__name__}")
    print(f"\n{len(tests)} 個測試全部通過")
