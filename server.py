import asyncio
import threading
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import main_model


# ============================================================
# FastAPI
# ============================================================

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 背景啟動骨架行為分析
# ============================================================

def run_skeleton_model():
    # main_model.main() 在鏡頭來源斷線時會直接結束（不是掛掉，是正常跑完），
    # 這裡包一層無限重試，斷線後每 5 秒自動重新嘗試連線，不用手動重啟 server.py
    while True:
        try:
            main_model.main()

        except Exception as e:
            print("❌ 骨架行為分析發生錯誤：", e)

        main_model.mark_offline()
        print("⏳ 鏡頭來源已中斷，5 秒後自動重新嘗試連線...")
        time.sleep(5)


@app.on_event("startup")
def startup_event():
    print("🚀 啟動骨架行為分析")

    model_thread = threading.Thread(
        target=run_skeleton_model,
        daemon=True
    )

    model_thread.start()


# ============================================================
# HTTP 測試
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Skeleton Behavior Server is running!"
    }


# ============================================================
# 取得目前最新骨架行為
# ============================================================

@app.get("/current")
def get_current_behavior():

    return main_model.latest_result


# ============================================================
# 即時影像（純攝影機畫面，不含骨架疊圖）
# 跟 yourpetapp 的 stream_server.py 用同一套 /ping、/video_feed 介面，
# 這樣 App 端不用改，只要把監視器伺服器指到這個 port 就好
# ============================================================

@app.get("/ping")
def ping():
    return {"service": "yourpetapp_monitor", "status": "ok"}


def generate_video_feed():
    while True:
        frame = main_model.latest_frame_jpeg

        if frame is None:
            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(0.03)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_video_feed(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================
# WebSocket
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ Expo 已連接骨架 WebSocket")

    try:
        while True:

            await websocket.send_json(
                main_model.latest_result
            )

            await asyncio.sleep(1)
    except WebSocketDisconnect:

        print("⚠️ Expo WebSocket 已斷線")