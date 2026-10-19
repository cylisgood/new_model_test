"""
串流中斷 / 恢復的整合測試：啟動真正的 server.py（走正式流程），讓它去讀一個本機模擬的串流，
中途把模擬串流關掉再重開，確認：
  1. 沒有設 SOURCE_MODE、只設 HUB_AI_STREAM 就會自動走串流模式（不會默默打開本機鏡頭）
  2. 串流中斷時，程式會發現、/current 立刻變 offline
  3. 串流恢復後，程式「自己」重新連上（不需要重啟）
  4. 恢復後分析仍然繼續產出結果
模擬的是 HTTP MJPEG（手機 IP Webcam 那種），不是 RTSP；Hub 實際重啟的情況沒辦法在這裡測。

需要 GPU / DeepLabCut，約 2~3 分鐘。用 127.0.0.1，不會對外開放任何 port。

用法（在專案根目錄）：
  venv\\Scripts\\python.exe evaluation\\test_stream_reconnect.py
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
FAKE_PORT = 8099
SERVER_PORT = 8001
FAKE_FPS = 15.0


# ---------------- 模擬串流伺服器 ----------------

def serve_fake_stream():
    import cv2

    video = REPO_DIR / "input" / "composite_side_by_side.mp4"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            cap = cv2.VideoCapture(str(video))
            nxt = time.time()
            try:
                while True:
                    ok, fr = cap.read()
                    if not ok:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    data = cv2.imencode(".jpg", cv2.resize(fr, (960, 360)), [int(cv2.IMWRITE_JPEG_QUALITY), 80])[1].tobytes()
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                    nxt += 1.0 / FAKE_FPS
                    time.sleep(max(0.0, nxt - time.time()))
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

    ThreadingHTTPServer(("127.0.0.1", FAKE_PORT), Handler).serve_forever()


# ---------------- 測試流程 ----------------

def kill_tree(proc):
    if proc and proc.poll() is None:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)


def read_log(path):
    return Path(path).read_text(encoding="utf-8", errors="replace") if Path(path).exists() else ""


def wait_for(predicate, timeout, what):
    start = time.time()
    while time.time() - start < timeout:
        if predicate():
            return time.time() - start
        time.sleep(0.5)
    raise AssertionError(f"等了 {timeout} 秒仍然沒有：{what}")


def http_json(path):
    with urllib.request.urlopen(f"http://127.0.0.1:{SERVER_PORT}{path}", timeout=5) as r:
        return json.loads(r.read().decode("utf-8"))


def unique_feed_frames(seconds=3.0):
    """讀 /video_feed 幾秒，回傳 (收到的幀數, 不重複的幀數)。來源斷線時畫面會停住，不重複的幀數就是 1。"""
    seen, total, buf = set(), 0, b""
    deadline = time.time() + seconds
    with urllib.request.urlopen(f"http://127.0.0.1:{SERVER_PORT}/video_feed", timeout=5) as r:
        while time.time() < deadline:
            chunk = r.read(65536)
            if not chunk:
                break
            buf += chunk
            parts = buf.split(b"--frame")
            buf = parts[-1]
            for part in parts[:-1]:
                if b"\xff\xd8" in part:
                    total += 1
                    seen.add(hashlib.md5(part).hexdigest())
    return total, len(seen)


def start_fake(log):
    return subprocess.Popen([sys.executable, "-u", str(Path(__file__)), "--serve"], cwd=REPO_DIR,
                            stdout=log, stderr=subprocess.STDOUT)


def main():
    scratch = Path(os.environ.get("TEMP", ".")) / "stream_reconnect_test"
    scratch.mkdir(parents=True, exist_ok=True)
    server_log = scratch / "server.log"
    fake_log_file = open(scratch / "fake.log", "w")

    env = dict(os.environ)
    env.pop("SOURCE_MODE", None)                      # 故意不設：要靠 HUB_AI_STREAM 自動選串流模式
    env.pop("RTSP_URL", None)
    env["HUB_AI_STREAM"] = f"http://127.0.0.1:{FAKE_PORT}/video"
    env["CLIP_DURATION_SECONDS"] = "5"
    env["PYTHONIOENCODING"] = "utf-8"

    fake = start_fake(fake_log_file)
    server = None
    results = []

    try:
        time.sleep(2)
        with open(server_log, "w", encoding="utf-8") as lf:
            server = subprocess.Popen(
                [sys.executable, "-u", "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", str(SERVER_PORT)],
                cwd=REPO_DIR, env=env, stdout=lf, stderr=subprocess.STDOUT)

        t = wait_for(lambda: "即時來源解析度" in read_log(server_log), 150, "server 連上串流並量完幀率")
        text = read_log(server_log)
        assert "來源模式： stream" in text, "沒設 SOURCE_MODE 卻沒有自動走串流模式"
        fps_line = next(l for l in text.splitlines() if l.startswith("FPS："))
        results.append(f"✅ 1. 只設 HUB_AI_STREAM 就自動走串流模式（{t:.0f} 秒後連上）；{fps_line}（模擬串流實際 {FAKE_FPS:g} fps）")

        total, uniq = unique_feed_frames()
        assert uniq > 1, f"串流正常時 /video_feed 的畫面應該在變，3 秒內只有 {uniq} 種"
        results.append(f"✅ 2. 串流正常：/video_feed 3 秒收到 {total} 幀、{uniq} 種不同畫面")

        # ---- 中斷 ----
        t_cut = time.time()
        kill_tree(fake)
        dt = wait_for(lambda: "即時來源中斷" in read_log(server_log), 60, "發現來源中斷")
        results.append(f"✅ 3. 來源中斷後 {dt:.0f} 秒內被發現（讀取逾時設 10 秒）")

        cur = http_json("/current")
        assert cur["status"] == "offline", cur
        results.append(f"✅ 4. /current 立刻變 offline：{cur['status']}")

        total, uniq = unique_feed_frames()
        assert uniq == 1, f"斷線後畫面應該停住，卻還有 {uniq} 種不同畫面"
        results.append("✅ 5. 斷線期間 /video_feed 畫面停住（沒有假裝還在更新）")

        assert server.poll() is None, "server 在斷線時結束了"
        time.sleep(5)  # 在斷線狀態多撐一會兒
        assert server.poll() is None
        # 「持續在重試」要靠下面第 7 項（後來真的自己恢復了）來證明；重試訊息只在嘗試失敗後才印，
        # 而每次嘗試最長會卡 10 秒（開啟逾時），所以這裡數訊息次數不能當證據
        results.append("✅ 6. 斷線期間 server 沒有結束")

        # ---- 恢復 ----
        analyzed_before = read_log(server_log).count("分析完一段影片")
        fake = start_fake(fake_log_file)
        dt = wait_for(lambda: "即時來源已恢復" in read_log(server_log), 60, "重新連上")
        results.append(f"✅ 7. 來源回來後 {dt:.0f} 秒內自動重新連上（總共中斷約 {time.time() - t_cut - dt:.0f} 秒），沒有重啟程式")

        time.sleep(2)
        total, uniq = unique_feed_frames()
        assert uniq > 1, "恢復後 /video_feed 的畫面應該又在變"
        results.append(f"✅ 8. 恢復後 /video_feed 又開始更新（3 秒 {uniq} 種不同畫面）")

        # ---- 分析繼續 ----
        dt = wait_for(lambda: read_log(server_log).count("分析完一段影片") > analyzed_before, 120, "恢復後仍有新的分析結果")
        results.append(f"✅ 9. 分析沒有受影響：恢復後仍產出新的分析結果（{dt:.0f} 秒內）")

        cur = http_json("/current")
        results.append(f"ℹ  /current 目前：status={cur['status']} behavior={cur['behavior']}")

        assert "Traceback" not in read_log(server_log), "server 日誌裡有 Traceback"
        results.append("✅ 10. server 日誌沒有 Traceback")

    except AssertionError as e:
        results.append(f"❌ 失敗：{e}")
        tail = "\n".join(read_log(server_log).splitlines()[-25:])
        results.append("--- server 日誌最後 25 行 ---\n" + tail)
    finally:
        kill_tree(server)
        kill_tree(fake)
        fake_log_file.close()

    print("\n".join(results))
    sys.exit(0 if not any(r.startswith("❌") for r in results) else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true", help="（內部使用）只啟動模擬串流伺服器")
    args = ap.parse_args()

    if args.serve:
        serve_fake_stream()
    else:
        main()
