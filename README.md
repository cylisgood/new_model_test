# petbehavior-dlc

使用 [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) SuperAnimal-Quadruped 模型做寵物（貓、狗）骨架偵測，再依骨架判斷行為（走路、站立、坐著、休息），並透過 FastAPI 提供即時結果給手機 App。

## 功能

- **骨架姿態估計**：SuperAnimal-Quadruped（`hrnet_w32` + `fasterrcnn_resnet50_fpn_v2`），39 個關鍵點。
- **多寵物追蹤**：以位置、體型、骨架特徵與顏色直方圖計算匹配分數，用匈牙利演算法做全域配對，轉身或靠近時較不易換 ID。
- **鏡頭晃動補償**：移動速度會扣除鏡頭本身的位移，避免手持或晃動的畫面被誤判成走路。
- **行為分類**：依「速度 ÷ 體長」、路徑效率與脊椎角度判斷。

  | 模型內部標籤 | 判斷方式 | API 輸出 |
  | --- | --- | --- |
  | `running`、`walking` | 移動速度達門檻 | `walking` |
  | `standing` | 低速、或原地晃動 | `standing` |
  | `sitting` | 靜止、脊椎偏直且後腳收起 | `sitting` |
  | `lying` | 靜止、脊椎接近水平 | `resting` |
- **三種輸入來源**：本機攝影機、RTSP / HTTP 串流、影片檔。
- **FastAPI 伺服器**：HTTP、WebSocket、MJPEG 即時影像；來源斷線時每 5 秒自動重連。

## 專案結構

| 檔案 | 說明 |
| --- | --- |
| `main_model.py` | 核心流程：DeepLabCut 推論、追蹤、行為分類、輸出標註影片與行為紀錄 |
| `server.py` | FastAPI 伺服器，在背景執行緒啟動 `main_model.main()` |
| `check_accuracy.py` | 對單支影片跑 SuperAnimal 推論，檢查關鍵點品質（`input/test.mp4`） |
| `check_accuracy_mypet.py` | 同上，改用自己的寵物影片 |
| `requirements.txt` | 相依套件（鎖定版本） |

`input/`（輸入影片）和 `output/`（輸出影片與紀錄）已加入 `.gitignore`，不在 repo 內，請自行建立。

## 環境需求

- Windows、Python 3.10
- NVIDIA GPU，`requirements.txt` 以 RTX 2060 + CUDA 12.1 為準

## 安裝

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 使用方式

### 1. 離線分析影片檔

把影片放進 `input/`，設定環境變數後執行：

```powershell
$env:SOURCE_MODE = "video"
$env:VIDEO_PATH = "input\your_video.mp4"
python main_model.py
```

整支影片只會送進 DeepLabCut 一次，逐幀追蹤並分類，輸出到 `output/`：

- `<影片名>_行為辨識影片.mp4`：畫上骨架與行為標籤的影片
- `<影片名>_行為紀錄.txt`：每隻寵物各行為的起訖時間

### 2. 啟動伺服器（即時來源）

```powershell
$env:SOURCE_MODE = "webcam"        # 或 stream
$env:CAMERA_INDEX = "0"            # webcam 時使用
$env:RTSP_URL = "rtsp://..."       # stream 時使用
uvicorn server:app --host 0.0.0.0 --port 8000
```

即時模式下，程式會持續錄影並每 `CLIP_DURATION_SECONDS` 秒（預設 12）切成一段，依序交給 DeepLabCut 分析。原因是 SuperAnimal-Quadruped 只提供整支影片的批次介面，沒有穩定的單幀即時推論。所以 `/video_feed` 是即時的，`/current` 與 `/ws` 的行為結果則是「約 N 秒前發生的事」。

### 環境變數

| 變數 | 預設值 | 說明 |
| --- | --- | --- |
| `SOURCE_MODE` | `webcam` | `webcam`、`stream` 或 `video` |
| `CAMERA_INDEX` | `0` | 攝影機編號 |
| `RTSP_URL` | 空 | `SOURCE_MODE=stream` 時必填 |
| `VIDEO_PATH` | `input/dog_3.mp4` | `SOURCE_MODE=video` 時的影片路徑 |
| `CLIP_DURATION_SECONDS` | `12` | 即時模式每段分析的秒數 |
| `PROJECT_DIR` | 程式所在資料夾 | `input/`、`output/` 的根目錄 |

## API

| 方法 | 路徑 | 說明 |
| --- | --- | --- |
| GET | `/` | 伺服器狀態 |
| GET | `/ping` | 健康檢查，回傳 `{"service": "yourpetapp_monitor", "status": "ok"}` |
| GET | `/current` | 目前最新的行為結果 |
| GET | `/video_feed` | MJPEG 即時影像（原始畫面，不含骨架） |
| WS | `/ws` | 每秒推送一次行為結果 |

`/current` 與 `/ws` 的回傳格式：

```json
{
  "status": "online",
  "behavior": "walking",
  "confidence": 0.82,
  "summary": { "walking": 1, "standing": 0, "sitting": 0, "resting": 0 }
}
```

- `status`：`online` 或 `offline`（來源中斷時）
- `behavior`：信心最高的那隻寵物的行為
- `confidence`：該寵物近期行為歷史中，最常出現的行為所占比例
- `summary`：目前各行為的寵物數量

CORS 目前設為允許所有來源（`*`），部署到公開網路前請收緊。

## 注意事項

- `check_accuracy.py` 與 `check_accuracy_mypet.py` 內的影片路徑是寫死的絕對路徑（`C:\Users\sam20\...`），換電腦或換影片時需自行修改。
- 行為判斷的門檻（速度、脊椎角度等）是針對測試影片調整的，不同鏡頭角度或體型可能需要微調，主要常數集中在 `main_model.py` 前半段與 `classify_action()`。
