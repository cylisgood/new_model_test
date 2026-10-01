import json
import os
import queue
import re
import sys
import cv2
import math
import threading
import time
import numpy as np
from pathlib import Path
from collections import deque, Counter
from scipy.optimize import linear_sum_assignment

import deeplabcut

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ============================================================
# 給 server.py /current、/ws 讀取的即時行為結果
# ============================================================

latest_result = {
    "status": "offline",
    "behavior": "unknown",
    "confidence": 0.0,
    "summary": {
        "walking": 0,
        "standing": 0,
        "sitting": 0,
        "resting": 0,
        "climbing": 0,
    },
}

# 給 server.py /video_feed 讀取的最新一幀畫面（純攝影機原始畫面，不含骨架疊圖）
latest_frame_jpeg = None


def update_latest_frame(frame):
    global latest_frame_jpeg

    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if ok:
        latest_frame_jpeg = jpg.tobytes()

# 把模型內部的行為標籤對應到前端看得懂的欄位（running 併入 walking，lying 併入 resting）
BEHAVIOR_LABEL_MAP = {
    "walking": "walking",
    "running": "walking",
    "standing": "standing",
    "sitting": "sitting",
    "lying": "resting",
}


def update_latest_result(tracks):
    """
    根據目前所有寵物的行為紀錄，更新 latest_result，
    供 FastAPI 的 /current、/ws 直接讀取。
    """
    global latest_result

    summary = {"walking": 0, "standing": 0, "sitting": 0, "resting": 0, "climbing": 0}
    top_behavior = "unknown"
    top_confidence = 0.0

    for track in tracks.values():
        action_history = track.get("action_history")

        if not action_history:
            continue

        action, count = Counter(action_history).most_common(1)[0]
        confidence = count / len(action_history)
        mapped_action = BEHAVIOR_LABEL_MAP.get(action)

        if mapped_action in summary:
            summary[mapped_action] += 1

        if confidence > top_confidence:
            top_confidence = confidence
            top_behavior = mapped_action or action

    latest_result = {
        "status": "online",
        "behavior": top_behavior,
        "confidence": round(top_confidence, 2),
        "summary": summary,
    }


def mark_offline():
    """鏡頭來源斷線、正在等待重試時呼叫，讓 /current 如實反映目前沒有資料可看。"""
    global latest_result

    latest_result = {
        "status": "offline",
        "behavior": "unknown",
        "confidence": 0.0,
        "summary": {
            "walking": 0,
            "standing": 0,
            "sitting": 0,
            "resting": 0,
            "climbing": 0,
        },
    }


# ============================================================
# 測試的新影片：只要改這裡
# ============================================================

PROJECT_DIR = Path(os.getenv("PROJECT_DIR", str(Path(__file__).resolve().parent)))

VIDEO_PATH = os.getenv("VIDEO_PATH", str(PROJECT_DIR / "input" / "dog_3.mp4"))
SOURCE_MODE = os.getenv("SOURCE_MODE", "webcam").strip().lower()
if SOURCE_MODE not in {"webcam", "stream", "video"}:
    SOURCE_MODE = ("stream")

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
STREAM_URL = os.getenv("RTSP_URL", "").strip()

# ============================================================
# 自動輸出路徑
# ============================================================

OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

video_path = Path(VIDEO_PATH)
video_name = video_path.stem

BEHAVIOR_LOG_PATH = OUTPUT_DIR / f"{video_name}_行為紀錄.txt"

runtime_source_mode = SOURCE_MODE
runtime_source_path = VIDEO_PATH
runtime_source_label = video_name


def sanitize_source_name(text):
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text))
    name = name.strip("._-")
    return name[:80] or "live_source"


def configure_runtime_paths(source_label):
    global video_name
    global BEHAVIOR_LOG_PATH
    global runtime_source_label

    runtime_source_label = sanitize_source_name(source_label)
    video_name = runtime_source_label

    BEHAVIOR_LOG_PATH = OUTPUT_DIR / f"{video_name}_行為紀錄.txt"


def resolve_input_source():
    """
    回傳 (mode, capture_source, source_label)
    - webcam: capture_source = 攝影機 index
    - stream: capture_source = RTSP / HTTP 串流 URL
    """
    if SOURCE_MODE == "webcam":
        return "webcam", CAMERA_INDEX, f"webcam_{CAMERA_INDEX}"

    if SOURCE_MODE == "stream":
        if not STREAM_URL:
            raise ValueError("SOURCE_MODE=stream 時請設定 STREAM_URL")
        return "stream", STREAM_URL, sanitize_source_name(STREAM_URL)

    if SOURCE_MODE == "video":
        video_file = Path(VIDEO_PATH)
        if not video_file.exists():
            raise FileNotFoundError(f"找不到影片：{VIDEO_PATH}")
        return "video", str(video_file), video_file.stem

    raise ValueError("SOURCE_MODE 只支援 webcam、stream 或 video")


def open_capture(capture_source, mode):
    if mode == "webcam":
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]

        for backend in backends:
            cap = cv2.VideoCapture(int(capture_source), backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            cap.release()

        return cv2.VideoCapture(int(capture_source))

    # RTSP 低延遲設定：優先丟掉緩衝、避免長時間排隊
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|max_delay;0|buffer_size;1024"
    cap = cv2.VideoCapture(str(capture_source), cv2.CAP_FFMPEG)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

# ============================================================
# DeepLabCut SuperAnimal-Quadruped 39 keypoints
# 順序必須跟模型實際輸出的順序一致（來自 video_inference_superanimal
# 產生的 .h5 欄位順序），dlc_json_frame_to_instances() 依賴這個順序
# 把 JSON 裡的 [x, y, score] 陣列對應回正確的部位名稱
# ============================================================

KEYPOINTS = [
    "nose", "upper_jaw", "lower_jaw", "mouth_end_right", "mouth_end_left",
    "right_eye", "right_earbase", "right_earend", "right_antler_base", "right_antler_end",
    "left_eye", "left_earbase", "left_earend", "left_antler_base", "left_antler_end",
    "neck_base", "neck_end", "throat_base", "throat_end",
    "back_base", "back_end", "back_middle",
    "tail_base", "tail_end",
    "front_left_thai", "front_left_knee", "front_left_paw",
    "front_right_thai", "front_right_knee", "front_right_paw",
    "back_left_paw", "back_left_thai", "back_right_thai",
    "back_left_knee", "back_right_knee", "back_right_paw",
    "belly_bottom", "body_middle_right", "body_middle_left",
]

SKELETON_LINES = [
    (KEYPOINTS.index("nose"), KEYPOINTS.index("upper_jaw")),
    (KEYPOINTS.index("left_eye"), KEYPOINTS.index("nose")),
    (KEYPOINTS.index("right_eye"), KEYPOINTS.index("nose")),
    (KEYPOINTS.index("left_eye"), KEYPOINTS.index("left_earbase")),
    (KEYPOINTS.index("right_eye"), KEYPOINTS.index("right_earbase")),
    (KEYPOINTS.index("nose"), KEYPOINTS.index("neck_base")),
    (KEYPOINTS.index("neck_base"), KEYPOINTS.index("neck_end")),
    (KEYPOINTS.index("neck_end"), KEYPOINTS.index("back_base")),
    (KEYPOINTS.index("back_base"), KEYPOINTS.index("back_middle")),
    (KEYPOINTS.index("back_middle"), KEYPOINTS.index("back_end")),
    (KEYPOINTS.index("back_end"), KEYPOINTS.index("tail_base")),
    (KEYPOINTS.index("tail_base"), KEYPOINTS.index("tail_end")),
    (KEYPOINTS.index("neck_base"), KEYPOINTS.index("front_left_thai")),
    (KEYPOINTS.index("front_left_thai"), KEYPOINTS.index("front_left_knee")),
    (KEYPOINTS.index("front_left_knee"), KEYPOINTS.index("front_left_paw")),
    (KEYPOINTS.index("neck_base"), KEYPOINTS.index("front_right_thai")),
    (KEYPOINTS.index("front_right_thai"), KEYPOINTS.index("front_right_knee")),
    (KEYPOINTS.index("front_right_knee"), KEYPOINTS.index("front_right_paw")),
    (KEYPOINTS.index("tail_base"), KEYPOINTS.index("back_left_thai")),
    (KEYPOINTS.index("back_left_thai"), KEYPOINTS.index("back_left_knee")),
    (KEYPOINTS.index("back_left_knee"), KEYPOINTS.index("back_left_paw")),
    (KEYPOINTS.index("tail_base"), KEYPOINTS.index("back_right_thai")),
    (KEYPOINTS.index("back_right_thai"), KEYPOINTS.index("back_right_knee")),
    (KEYPOINTS.index("back_right_knee"), KEYPOINTS.index("back_right_paw")),
]

# 行為判斷邏輯用到的固定部位，集中定義索引，不再像 MMPose 版那樣把數字寫死在函式裡
# neck_base 只有約 68% 的幀有效且位置會跳，脊椎前端改用 97% 有效的 throat_base
NECK_IDX = KEYPOINTS.index("throat_base")
TAIL_IDX = KEYPOINTS.index("tail_base")
LEFT_SHOULDER_IDX = KEYPOINTS.index("front_left_thai")
RIGHT_SHOULDER_IDX = KEYPOINTS.index("front_right_thai")
LEFT_HIP_IDX = KEYPOINTS.index("back_left_thai")
RIGHT_HIP_IDX = KEYPOINTS.index("back_right_thai")
LEFT_FRONT_PAW_IDX = KEYPOINTS.index("front_left_paw")
RIGHT_FRONT_PAW_IDX = KEYPOINTS.index("front_right_paw")
LEFT_BACK_PAW_IDX = KEYPOINTS.index("back_left_paw")
RIGHT_BACK_PAW_IDX = KEYPOINTS.index("back_right_paw")

# 實測 SuperAnimal-Quadruped 的 neck_base / neck_end / back_base 只有約 68% 的幀信心度夠，
# 且位置會在肩頸之間跳動；身體中心改用這幾個 94% 以上都穩定有效的軀幹點取平均。
BODY_CENTER_IDXS = [
    KEYPOINTS.index(name) for name in (
        "back_middle", "back_end", "tail_base",
        "front_left_thai", "front_right_thai", "back_left_thai",
        "body_middle_right", "body_middle_left",
    )
]

# ============================================================
# 參數設定
# ============================================================

# 骨架點信心門檻
CONF_THRES = 0.35

# Skeleton-based Persistent Tracking 參數
MAX_MISSING_FRAMES = 90  # 寵物消失幾幀內仍保留原 ID；30FPS 約 3 秒
MATCH_THRES = 2.0  # 越大越容易沿用舊 ID，越小越嚴格

# 過濾假偵測 / 重複偵測，避免同一隻寵物被切成好幾個 ID
MIN_VALID_KEYPOINTS = 16       # 有效關鍵點少於這個數量（約 40%）的偵測不當作寵物（玩具、局部物體）
DUPLICATE_IOU_THRES = 0.5      # 同一幀兩個偵測框重疊超過這個比例，視為同一隻，只留關鍵點較多的
TRACK_MIN_HITS = 5             # 新出現的追蹤要連續被配對到幾幀才算數，之後才會顯示 / 記錄行為
TENTATIVE_MAX_MISSING = 3      # 還沒確認的追蹤，超過幾幀沒配對到就丟掉

# --- 新增：外觀(顏色)特徵權重與 bbox 相關參數 ---
COLOR_HIST_BINS = 16          # HSV 色調直方圖的 bin 數
BBOX_MARGIN_RATIO = 0.25      # 由 bbox 過濾離群關鍵點時的外擴比例
FALLBACK_BBOX_MARGIN = 0.15   # 無 bbox 時，用關鍵點自行估計 bbox 的外擴比例

# 配對分數權重（總和為 1，越低分越像同一隻寵物）
W_CENTER = 0.40
W_BODYSIZE = 0.15
W_FEATURE = 0.10
W_COLOR = 0.35

# 坐姿要求後腳收起來：髖部到後腳掌的垂直距離 ÷ 身長。
# 實測坐著的貓 0.27~0.39，站著/踱步/正面走路都在 0.52 以上，取中間空隙 0.45。
SITTING_MAX_HIP_TO_BACK_PAW = 0.45

# --- 新增：行為判斷相關參數 ---
PATH_EFFICIENCY_THRES = 0.35  # 淨位移 / 總路徑長，低於此值視為「原地抖動」而非真的移動
CAMERA_COMP_ENABLE = True      # 是否啟用鏡頭晃動補償

# 多寵物顏色
PET_COLORS = [
    (255, 0, 0),  # 藍色
    (0, 255, 0),  # 綠色
    (0, 0, 255),  # 紅色
    (255, 255, 0),  # 青色
    (255, 0, 255),  # 紫色
    (0, 255, 255),  # 黃色
]


# ============================================================
# 基礎工具函式
# ============================================================

def distance(p1, p2):
    if p1 is None or p2 is None:
        return None

    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def format_time(seconds):
    seconds = int(seconds)
    minutes = seconds // 60
    sec = seconds % 60
    return f"{minutes:02d}:{sec:02d}"


def update_behavior_segments(behavior_segments, active_segments, pet_id, action, frame_id):
    """
    根據每一幀的 action 更新行為區段紀錄。
    """

    if pet_id not in behavior_segments:
        behavior_segments[pet_id] = []

    if pet_id not in active_segments:
        active_segments[pet_id] = {
            "action": action,
            "start_frame": frame_id,
            "end_frame": frame_id
        }
        return

    current_segment = active_segments[pet_id]

    if current_segment["action"] == action:
        current_segment["end_frame"] = frame_id
        return

    behavior_segments[pet_id].append({
        "action": current_segment["action"],
        "start_frame": current_segment["start_frame"],
        "end_frame": current_segment["end_frame"]
    })

    active_segments[pet_id] = {
        "action": action,
        "start_frame": frame_id,
        "end_frame": frame_id
    }


def write_behavior_log(behavior_segments, active_segments, fps, video_name, output_path):
    """
    將所有寵物的行為區段輸出成文字檔。
    """

    for pet_id, segment in active_segments.items():
        if pet_id not in behavior_segments:
            behavior_segments[pet_id] = []

        behavior_segments[pet_id].append({
            "action": segment["action"],
            "start_frame": segment["start_frame"],
            "end_frame": segment["end_frame"]
        })

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Pet Behavior Log\n")
        f.write("==============================\n")
        f.write(f"Video: {video_name}\n")
        f.write(f"FPS: {fps}\n")
        f.write("==============================\n\n")

        for pet_id in sorted(behavior_segments.keys()):
            f.write(f"Pet {pet_id}\n")
            f.write("------------------------------\n")

            for segment in behavior_segments[pet_id]:
                start_time = segment["start_frame"] / fps
                end_time = segment["end_frame"] / fps
                action = segment["action"]

                start_text = format_time(start_time)
                end_text = format_time(end_time)

                f.write(f"{start_text} - {end_text} | {action}\n")

            f.write("\n")

    print("✅ 行為紀錄文字檔輸出完成：", output_path)


def get_point(kpts, scores, idx):
    if kpts is None:
        return None

    if idx >= len(kpts):
        return None

    if scores is not None and idx < len(scores):
        if scores[idx] < CONF_THRES:
            return None

    try:
        x = int(kpts[idx][0])
        y = int(kpts[idx][1])
        return x, y
    except Exception:
        return None


def avg_points(points):
    points = [p for p in points if p is not None]

    if not points:
        return None

    x = sum(p[0] for p in points) / len(points)
    y = sum(p[1] for p in points) / len(points)

    return int(x), int(y)


def body_center(kpts, scores):
    torso = [get_point(kpts, scores, idx) for idx in BODY_CENTER_IDXS]
    torso = [p for p in torso if p is not None]

    if len(torso) >= 3:
        return (
            int(sum(p[0] for p in torso) / len(torso)),
            int(sum(p[1] for p in torso) / len(torso)),
        )

    neck = get_point(kpts, scores, NECK_IDX)
    tail = get_point(kpts, scores, TAIL_IDX)

    if neck is not None and tail is not None:
        return (
            int((neck[0] + tail[0]) / 2),
            int((neck[1] + tail[1]) / 2)
        )

    return neck or tail


# ============================================================
# 【新增】bbox 工具：取得/估計每隻動物的邊界框
# 用途：
#   1. 提供比 neck-tail 距離更穩定的體型量測（不受轉身影響）
#   2. 用來裁切畫面計算外觀顏色特徵，強化多寵物 ID 配對
#   3. 過濾掉不屬於這隻動物、卻混進來的離群關鍵點
# ============================================================

def normalize_bbox(raw_bbox, frame_w, frame_h):
    """
    嘗試把各種可能的 bbox 格式統一成 (x1, y1, x2, y2)。
    支援 [x1,y1,x2,y2] 或 [x1,y1,x2,y2,score] 或 [[x1,y1,x2,y2]] 這幾種常見寫法。
    """
    if raw_bbox is None:
        return None

    try:
        b = raw_bbox

        # 有些格式會多包一層 list，例如 [[x1,y1,x2,y2]]
        if len(b) > 0 and isinstance(b[0], (list, tuple)):
            b = b[0]

        if len(b) < 4:
            return None

        x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])

        if x2 < x1:
            x1, x2 = x2, x1

        if y2 < y1:
            y1, y2 = y2, y1

        x1 = max(0, min(frame_w - 1, x1))
        x2 = max(0, min(frame_w - 1, x2))
        y1 = max(0, min(frame_h - 1, y1))
        y2 = max(0, min(frame_h - 1, y2))

        if x2 - x1 < 4 or y2 - y1 < 4:
            return None

        return x1, y1, x2, y2
    except Exception:
        return None


def bbox_from_keypoints(kpts, scores, frame_w, frame_h, margin_ratio=FALLBACK_BBOX_MARGIN):
    """
    當 JSON 沒有提供 bbox 時，用有效關鍵點的最小外接框往外擴一點當作 bbox。
    """
    pts = []

    for i in range(len(KEYPOINTS)):
        p = get_point(kpts, scores, i)
        if p is not None:
            pts.append(p)

    if len(pts) < 2:
        return None

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    w = x2 - x1
    h = y2 - y1

    x1 -= w * margin_ratio
    x2 += w * margin_ratio
    y1 -= h * margin_ratio
    y2 += h * margin_ratio

    x1 = max(0, min(frame_w - 1, x1))
    x2 = max(0, min(frame_w - 1, x2))
    y1 = max(0, min(frame_h - 1, y1))
    y2 = max(0, min(frame_h - 1, y2))

    if x2 - x1 < 4 or y2 - y1 < 4:
        return None

    return x1, y1, x2, y2


def bbox_diagonal(bbox):
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    return math.hypot(x2 - x1, y2 - y1)


def filter_keypoints_by_bbox(kpts, scores, bbox, margin_ratio=BBOX_MARGIN_RATIO):
    """
    【解決 keypoint 混淆問題】
    如果某個關鍵點的座標遠遠超出這隻動物自己的 bbox（外擴一定比例後),
    很可能是姿態估計把「另一隻動物」的關鍵點錯配過來了，這裡直接把它視為無效。
    回傳新的 scores（分數被清成 0 的點，之後就不會被使用/繪製）。
    """
    if bbox is None or kpts is None or scores is None:
        return scores

    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    ex1 = x1 - w * margin_ratio
    ex2 = x2 + w * margin_ratio
    ey1 = y1 - h * margin_ratio
    ey2 = y2 + h * margin_ratio

    new_scores = list(scores)

    for i in range(min(len(kpts), len(new_scores))):
        if new_scores[i] < CONF_THRES:
            continue

        try:
            x, y = kpts[i][0], kpts[i][1]
        except Exception:
            continue

        if x < ex1 or x > ex2 or y < ey1 or y > ey2:
            new_scores[i] = 0.0  # 標記為無效，等同於信心不足

    return new_scores


def compute_color_hist(frame, bbox):
    """
    計算 bbox 區域的 HSV 色調直方圖，作為這隻動物的「外觀特徵」。
    兩隻毛色差異大的寵物（例如黃狗 vs 黑白狗）用這個特徵配對，
    可以大幅降低單靠位置/體型判斷時，轉身或靠太近造成的 ID 切換問題。
    """
    if frame is None or bbox is None:
        return None

    x1, y1, x2, y2 = [int(v) for v in bbox]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    if x2 - x1 < 4 or y2 - y1 < 4:
        return None

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv], [0, 1], None,
        [COLOR_HIST_BINS, COLOR_HIST_BINS],
        [0, 180, 0, 256]
    )

    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return hist


def compare_color_hist(hist1, hist2):
    """
    回傳兩個顏色直方圖的「差異分數」，0 代表完全相同，1 代表完全不同。
    """
    if hist1 is None or hist2 is None:
        return 0.5  # 沒有顏色資訊時，給中性分數，不影響太多

    corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    corr = max(-1.0, min(1.0, corr))

    return (1.0 - corr) / 2.0


def extract_instance(instance):
    """
    從 MMPose 的一個 instance dict 中取出 keypoints / scores / bbox。
    （新增：一併取出 bbox，供體型量測、顏色特徵、離群點過濾使用）
    """
    if not isinstance(instance, dict):
        return None, None, None

    kpts = instance.get("keypoints", None)
    scores = instance.get("keypoint_scores", None)
    raw_bbox = instance.get("bbox", None)

    if kpts is None:
        return None, None, None

    # 有些 JSON 會多包一層
    if len(kpts) > 0 and isinstance(kpts[0], list):
        if len(kpts[0]) > 0 and isinstance(kpts[0][0], list):
            kpts = kpts[0]

    if scores is not None:
        if len(scores) > 0 and isinstance(scores[0], list):
            scores = scores[0]

    # 如果 keypoints 是 [x, y, score]
    if scores is None:
        clean_kpts = []
        clean_scores = []

        for p in kpts:
            if len(p) >= 3:
                clean_kpts.append([p[0], p[1]])
                clean_scores.append(p[2])
            elif len(p) >= 2:
                clean_kpts.append([p[0], p[1]])
                clean_scores.append(1.0)

        kpts = clean_kpts
        scores = clean_scores

    return kpts, scores, raw_bbox


# ============================================================
# DeepLabCut 批次分析：把錄好的一小段影片丟給 SuperAnimal-Quadruped，
# 解析回傳的 JSON，轉成跟舊版 MMPose 相同的 instance dict 格式，
# 這樣 process_behavior_frame() 以下的追蹤/行為判斷邏輯完全不用改
# ============================================================

DLC_SUPERANIMAL_NAME = "superanimal_quadruped"
DLC_MODEL_NAME = "hrnet_w32"
DLC_DETECTOR_NAME = "fasterrcnn_resnet50_fpn_v2"


def dlc_json_frame_to_instances(frame_data):
    """
    把 DeepLabCut 一幀的 {"bodyparts": [[[x,y,score], ...39個...], ...每隻動物...]}
    轉成 process_behavior_frame() 看得懂的 instance dict list。
    信心值全部 <0 代表這個「動物欄位」這一幀根本沒偵測到東西，直接跳過。
    """
    instances = []

    for individual in frame_data.get("bodyparts", []):
        scores = [p[2] for p in individual]

        if all(s < 0 for s in scores):
            continue

        kpts = [[p[0], p[1]] for p in individual]

        instances.append({
            "keypoints": kpts,
            "keypoint_scores": scores,
            "bbox": None,
        })

    return instances


def run_dlc_clip_inference(clip_path):
    """
    對一小段影片跑 SuperAnimal-Quadruped，回傳每一幀的 instances list。
    跑完會把 DeepLabCut 產生的中間檔（.h5/.json/.pickle/標註影片）一併清掉，
    避免這個長駐程式一直跑下去，硬碟被中間檔塞滿。
    """
    clip_path = Path(clip_path)

    deeplabcut.video_inference_superanimal(
        [str(clip_path)],
        DLC_SUPERANIMAL_NAME,
        model_name=DLC_MODEL_NAME,
        detector_name=DLC_DETECTOR_NAME,
        video_adapt=False,
    )

    json_candidates = list(clip_path.parent.glob(f"{clip_path.stem}_{DLC_SUPERANIMAL_NAME}_*_before_adapt.json"))

    frames = []

    if json_candidates:
        with open(json_candidates[0], "r", encoding="utf-8") as f:
            frames_data = json.load(f)

        frames = [dlc_json_frame_to_instances(frame_data) for frame_data in frames_data]

    # 只清掉 DeepLabCut 自己產生的中間檔（.h5/.pickle/標註影片等），
    # 原始影片檔先留著，呼叫端還要讀回真實畫面做顏色直方圖／鏡頭晃動補償
    for extra_file in clip_path.parent.glob(f"{clip_path.stem}_{DLC_SUPERANIMAL_NAME}_*"):
        extra_file.unlink(missing_ok=True)

    return frames


def count_valid_keypoints(kpts, scores):
    return sum(1 for i in range(len(KEYPOINTS)) if get_point(kpts, scores, i) is not None)


def bbox_iou(a, b):
    if a is None or b is None:
        return 0.0

    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def drop_false_and_duplicate_detections(detections):
    """
    偵測器常見兩種雜訊：
    1. 玩具、局部物體被當成寵物 —— 有效關鍵點很少，直接濾掉
    2. 同一隻寵物被框兩次 —— 框幾乎重疊，只留關鍵點較多的那個
    """
    candidates = [
        d for d in detections
        if count_valid_keypoints(d["kpts"], d["scores"]) >= MIN_VALID_KEYPOINTS
    ]
    candidates.sort(key=lambda d: count_valid_keypoints(d["kpts"], d["scores"]), reverse=True)

    kept = []

    for det in candidates:
        if any(bbox_iou(det["bbox"], k["bbox"]) > DUPLICATE_IOU_THRES for k in kept):
            continue
        kept.append(det)

    return kept


def process_behavior_frame(
    frame,
    frame_id,
    instances,
    tracks,
    behavior_segments,
    active_segments,
    frame_w,
    frame_h,
    camera_motion_cache,
    prev_gray,
):
    """
    把單一 frame 的 detection / tracking / 行為判斷流程集中處理。
    """
    frame_actions = []

    for pet_id in tracks:
        tracks[pet_id]["missing_frames"] += 1

    detections = []

    for instance in instances:
        kpts, scores, raw_bbox = extract_instance(instance)

        if kpts is None:
            continue

        det = extract_detection_features(kpts, scores, raw_bbox, frame, frame_w, frame_h)

        if det["center"] is None:
            continue

        detections.append(det)

    detections = drop_false_and_duplicate_detections(detections)

    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if CAMERA_COMP_ENABLE:
        exclude_boxes = [d.get("bbox") for d in detections]
        cam_dx, cam_dy = estimate_camera_motion(prev_gray, curr_gray, exclude_boxes)
        camera_motion_cache[frame_id] = (cam_dx, cam_dy)
    else:
        camera_motion_cache[frame_id] = (0.0, 0.0)

    prev_gray = curr_gray

    assigned_ids = match_detections_to_tracks(detections, tracks)

    next_pet_id = max(tracks.keys(), default=0) + 1

    for det, pet_id in zip(detections, assigned_ids):
        if pet_id is None:
            pet_id = next_pet_id
            next_pet_id += 1

        if pet_id not in tracks:
            tracks[pet_id] = create_new_track(det, frame_id)
        else:
            update_track(tracks[pet_id], det, frame_id)

        if not tracks[pet_id]["confirmed"]:
            continue

        raw_action = classify_action(
            tracks[pet_id]["history"],
            frame_w,
            frame_h,
            det["kpts"],
            det["scores"],
            camera_motion_cache=camera_motion_cache,
        )

        tracks[pet_id]["action_history"].append(raw_action)

        action = Counter(tracks[pet_id]["action_history"]).most_common(1)[0][0]
        color = PET_COLORS[(pet_id - 1) % len(PET_COLORS)]

        frame_actions.append((pet_id, action, color))

        update_behavior_segments(
            behavior_segments,
            active_segments,
            pet_id,
            action,
            frame_id,
        )

        draw_pose(frame, det["kpts"], det["scores"], color=color)

        label_center = det["center"]

        if label_center is not None:
            cv2.putText(
                frame,
                f"Pet {pet_id}",
                (label_center[0] + 10, label_center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.75,
                color,
                3,
            )

    dead_ids = [
        pet_id for pet_id, track in tracks.items()
        if track["missing_frames"] > (MAX_MISSING_FRAMES if track["confirmed"] else TENTATIVE_MAX_MISSING)
    ]

    for pet_id in dead_ids:
        del tracks[pet_id]

    old_frame_cutoff = frame_id - MAX_MISSING_FRAMES - 30
    if old_frame_cutoff > 0 and old_frame_cutoff in camera_motion_cache:
        for f_old in list(camera_motion_cache.keys()):
            if f_old < old_frame_cutoff:
                del camera_motion_cache[f_old]

    return frame_actions, detections, prev_gray


def get_body_axis(kpts, scores):
    neck = get_point(kpts, scores, NECK_IDX)
    tail = get_point(kpts, scores, TAIL_IDX)

    if neck is None:
        left_shoulder = get_point(kpts, scores, LEFT_SHOULDER_IDX)
        right_shoulder = get_point(kpts, scores, RIGHT_SHOULDER_IDX)
        neck = avg_points([left_shoulder, right_shoulder])

    if tail is None:
        left_hip = get_point(kpts, scores, LEFT_HIP_IDX)
        right_hip = get_point(kpts, scores, RIGHT_HIP_IDX)
        tail = avg_points([left_hip, right_hip])

    body_len = distance(neck, tail) if (neck and tail) else None

    return neck, tail, body_len


# ============================================================
# EMA 平滑中心點
# ============================================================

def update_ema_center(raw_center, prev_ema, alpha=0.4):
    if raw_center is None:
        return prev_ema

    if prev_ema is None:
        return raw_center

    x = int(alpha * raw_center[0] + (1 - alpha) * prev_ema[0])
    y = int(alpha * raw_center[1] + (1 - alpha) * prev_ema[1])

    return x, y


# ============================================================
# 姿態特徵
# ============================================================

def get_posture_features(kpts, scores, body_len):
    features = {}

    if body_len is None or body_len < 1:
        return features

    neck, tail, _ = get_body_axis(kpts, scores)

    left_hip = get_point(kpts, scores, LEFT_HIP_IDX)
    right_hip = get_point(kpts, scores, RIGHT_HIP_IDX)
    left_shoulder = get_point(kpts, scores, LEFT_SHOULDER_IDX)
    right_shoulder = get_point(kpts, scores, RIGHT_SHOULDER_IDX)
    left_front_paw = get_point(kpts, scores, LEFT_FRONT_PAW_IDX)
    right_front_paw = get_point(kpts, scores, RIGHT_FRONT_PAW_IDX)

    if neck and tail:
        dy = abs(neck[1] - tail[1])
        dx = abs(neck[0] - tail[0]) + 1e-6
        features["spine_vertical_ratio"] = dy / (dy + dx)

    hip_center = avg_points([left_hip, right_hip])
    shoulder_center = avg_points([left_shoulder, right_shoulder])

    if hip_center and shoulder_center:
        features["hip_above_shoulder"] = (shoulder_center[1] - hip_center[1]) / body_len

    if left_front_paw and right_front_paw:
        features["paw_spread"] = abs(left_front_paw[0] - right_front_paw[0]) / body_len

    return features


def hip_to_back_paw_ratio(kpts, scores, body_len):
    if body_len is None or body_len < 1:
        return None

    hip = avg_points([
        get_point(kpts, scores, LEFT_HIP_IDX),
        get_point(kpts, scores, RIGHT_HIP_IDX),
    ])
    back_paw = avg_points([
        get_point(kpts, scores, LEFT_BACK_PAW_IDX),
        get_point(kpts, scores, RIGHT_BACK_PAW_IDX),
    ])

    if hip is None or back_paw is None:
        return None

    return (back_paw[1] - hip[1]) / body_len


def get_smoothed_posture_features(hist_list, kpts, scores, body_len):
    """
    脊椎角度單幀會因為關鍵點抖動，在 sitting/standing 的門檻附近來回跳，
    改取最近這段歷史（約 1 秒）的中位數，讓姿態判斷穩定。
    """
    features = get_posture_features(kpts, scores, body_len)

    if not features:
        return features

    ratios = []

    for h in hist_list:
        neck, tail = h.get("neck"), h.get("tail")

        if neck and tail:
            dy = abs(neck[1] - tail[1])
            dx = abs(neck[0] - tail[0]) + 1e-6
            ratios.append(dy / (dy + dx))

    if ratios:
        features["spine_vertical_ratio"] = float(np.median(ratios))

    leg_ratios = [h["hip_to_back_paw"] for h in hist_list if h.get("hip_to_back_paw") is not None]

    if leg_ratios:
        features["hip_to_back_paw"] = float(np.median(leg_ratios))

    return features


def classify_static_posture(features):
    spine_vr = features.get("spine_vertical_ratio", None)
    hip_to_paw = features.get("hip_to_back_paw", None)

    if spine_vr is not None and spine_vr < 0.25:
        return "lying"

    if spine_vr is not None and spine_vr > 0.45:
        # 貓正面朝鏡頭走時脊椎在畫面上會被壓成直的，光看脊椎角度會誤判成坐；
        # 真的坐著時後腳是收起來的，髖部離後腳掌很近，腿伸直就不算坐
        if hip_to_paw is not None and hip_to_paw >= SITTING_MAX_HIP_TO_BACK_PAW:
            return "standing"
        return "sitting"

    return "standing"


# ============================================================
# Skeleton-based Persistent Tracking（改進版）
# 改進重點：
#   1. body_size 優先用 bbox 對角線（不受轉身/姿態影響），沒有 bbox 才退回 neck-tail 距離
#   2. 加入顏色直方圖(外觀)特徵，大幅降低轉身、動物靠太近時的 ID 切換問題
#   3. 用 Hungarian algorithm 做全域最佳配對，取代原本逐一貪婪配對
# ============================================================

def extract_detection_features(kpts, scores, raw_bbox, frame, frame_w, frame_h):
    """
    將目前偵測到的一隻寵物整理成 tracking 可用資料。
    """
    # 1. 先決定這隻動物的 bbox（JSON 提供的優先，沒有就用關鍵點估計）
    bbox = normalize_bbox(raw_bbox, frame_w, frame_h)

    if bbox is None:
        bbox = bbox_from_keypoints(kpts, scores, frame_w, frame_h)

    # 2.【解決 keypoint 混淆】用 bbox 過濾掉明顯不屬於這隻動物的離群關鍵點
    if bbox is not None:
        scores = filter_keypoints_by_bbox(kpts, scores, bbox)

    neck, tail, body_len = get_body_axis(kpts, scores)
    center = body_center(kpts, scores)

    if center is None and bbox is not None:
        # 連 neck/tail 都拿不到時，退而求其次用 bbox 中心
        x1, y1, x2, y2 = bbox
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

    left_front = get_point(kpts, scores, LEFT_FRONT_PAW_IDX)
    right_front = get_point(kpts, scores, RIGHT_FRONT_PAW_IDX)
    left_back = get_point(kpts, scores, LEFT_BACK_PAW_IDX)
    right_back = get_point(kpts, scores, RIGHT_BACK_PAW_IDX)

    front_paw = avg_points([left_front, right_front])
    back_paw = avg_points([left_back, right_back])

    features = get_posture_features(kpts, scores, body_len)

    # body_size：優先用 bbox 對角線，這個量測不會因為動物轉身、伸展而暴衝
    body_size = bbox_diagonal(bbox) if bbox is not None else body_len

    color_hist = compute_color_hist(frame, bbox) if bbox is not None else None

    return {
        "kpts": kpts,
        "scores": scores,
        "center": center,
        "neck": neck,
        "tail": tail,
        "body_len": body_len,
        "body_size": body_size,
        "bbox": bbox,
        "color_hist": color_hist,
        "front_paw": front_paw,
        "back_paw": back_paw,
        "hip_to_back_paw": hip_to_back_paw_ratio(kpts, scores, body_len),
        "features": features
    }


def predict_center(track):
    last_center = track.get("last_center")
    velocity = track.get("velocity", (0, 0))

    if last_center is None:
        return None

    return (
        int(last_center[0] + velocity[0]),
        int(last_center[1] + velocity[1])
    )


def compute_match_score(det, track):
    """
    分數越低，代表越像同一隻寵物。
    綜合考慮：
    1. 預測中心點距離
    2. body_size（bbox 對角線）相似度 —— 比 neck-tail 距離更抗轉身
    3. skeleton posture feature 相似度
    4. 外觀顏色相似度 —— 對兩隻毛色差異大的寵物特別有效
    """
    if det["center"] is None or track.get("last_center") is None:
        return float("inf")

    body_size = det.get("body_size") or track.get("last_body_size")

    if body_size is None or body_size < 1:
        body_size = 150

    predicted_center = predict_center(track)

    if predicted_center is None:
        predicted_center = track["last_center"]

    center_dist = distance(det["center"], predicted_center)

    if center_dist is None:
        return float("inf")

    center_score = center_dist / body_size

    # body_size 相似度
    body_size_score = 0

    if det.get("body_size") is not None and track.get("last_body_size") is not None:
        if track["last_body_size"] > 1:
            body_size_score = abs(det["body_size"] - track["last_body_size"]) / track["last_body_size"]

    # posture feature 相似度
    feature_score = 0

    det_features = det.get("features", {})
    track_features = track.get("last_features", {})

    common_keys = set(det_features.keys()) & set(track_features.keys())

    if common_keys:
        diffs = [abs(det_features[key] - track_features[key]) for key in common_keys]
        feature_score = sum(diffs) / len(diffs)

    # 外觀顏色相似度（新增，權重最高，最能區分毛色不同的寵物）
    color_score = compare_color_hist(
        det.get("color_hist"),
        track.get("last_color_hist")
    )

    score = (
        center_score * W_CENTER +
        body_size_score * W_BODYSIZE +
        feature_score * W_FEATURE +
        color_score * W_COLOR
    )

    return score


def create_new_track(det, frame_id):
    history = deque(maxlen=30)
    action_history = deque(maxlen=10)

    history.append({
        "center": det["center"],
        "neck": det["neck"],
        "tail": det["tail"],
        "body_len": det["body_len"],
        "front_paw": det["front_paw"],
        "back_paw": det["back_paw"],
        "hip_to_back_paw": det.get("hip_to_back_paw"),
        "frame_id": frame_id
    })

    return {
        "history": history,
        "action_history": action_history,

        "last_center": det["center"],
        "ema_center": det["center"],
        "last_body_len": det["body_len"],
        "last_body_size": det["body_size"],
        "last_color_hist": det["color_hist"],
        "last_features": det["features"],

        "velocity": (0, 0),
        "missing_frames": 0,
        "active": True,
        "hits": 1,
        "confirmed": TRACK_MIN_HITS <= 1,
    }


def update_track(track, det, frame_id):
    old_center = track.get("last_center")
    new_center = det["center"]

    if old_center is not None and new_center is not None:
        vx = new_center[0] - old_center[0]
        vy = new_center[1] - old_center[1]
        track["velocity"] = (vx, vy)

    track["last_center"] = new_center
    track["last_body_len"] = det["body_len"]
    track["last_body_size"] = det["body_size"] if det["body_size"] is not None else track.get("last_body_size")

    # 顏色直方圖用平滑更新，避免單一幀光線/角度變化造成誤判
    if det.get("color_hist") is not None:
        if track.get("last_color_hist") is not None:
            track["last_color_hist"] = 0.7 * track["last_color_hist"] + 0.3 * det["color_hist"]
        else:
            track["last_color_hist"] = det["color_hist"]

    track["last_features"] = det["features"]
    track["missing_frames"] = 0

    track["hits"] = track.get("hits", 0) + 1
    if track["hits"] >= TRACK_MIN_HITS:
        track["confirmed"] = True

    track["ema_center"] = update_ema_center(
        new_center,
        track.get("ema_center"),
        alpha=0.4
    )

    track["history"].append({
        "center": track["ema_center"],
        "neck": det["neck"],
        "tail": det["tail"],
        "body_len": det["body_len"],
        "front_paw": det["front_paw"],
        "back_paw": det["back_paw"],
        "hip_to_back_paw": det.get("hip_to_back_paw"),
        "frame_id": frame_id
    })


def match_detections_to_tracks(detections, tracks):
    """
    【改進】用 Hungarian algorithm 做全域最佳配對，取代原本「誰先搶到算誰的」貪婪配對。
    這樣當兩隻寵物距離很近、彼此都可能像對方的 track 時，也能找到全局最合理的配對，
    大幅降低 ID 互換 (ID swap) 與不必要的新 ID 生成。

    回傳： list，長度與 detections 相同，每個元素是配對到的 pet_id（沒配對到就是 None）
    """
    valid_track_ids = [
        pid for pid, t in tracks.items()
        if t["missing_frames"] <= MAX_MISSING_FRAMES
    ]

    n_det = len(detections)
    n_trk = len(valid_track_ids)

    assigned = [None] * n_det

    if n_det == 0 or n_trk == 0:
        return assigned

    # 建立成本矩陣
    cost_matrix = np.full((n_det, n_trk), 1e6, dtype=np.float64)

    for i, det in enumerate(detections):
        for j, pid in enumerate(valid_track_ids):
            score = compute_match_score(det, tracks[pid])
            if math.isfinite(score):
                cost_matrix[i, j] = score

    row_idx, col_idx = linear_sum_assignment(cost_matrix)

    for r, c in zip(row_idx, col_idx):
        score = cost_matrix[r, c]
        if score < MATCH_THRES:
            assigned[r] = valid_track_ids[c]

    return assigned


# ============================================================
# 【新增】鏡頭晃動補償
# 目的：如果拍攝時鏡頭本身有輕微移動/手震，畫面中所有關鍵點都會一起偏移，
#      這會讓靜止不動的寵物被誤判成 walking/running。
#      這裡用畫面「背景」的 optical flow 估計鏡頭本身的移動量，
#      之後計算寵物移動速度時，把這個鏡頭位移量扣掉。
# ============================================================

def estimate_camera_motion(prev_gray, curr_gray, exclude_boxes):
    """
    回傳 (dx, dy)：代表鏡頭從 prev frame 到 curr frame 的估計位移。
    exclude_boxes: 目前幀所有動物的 bbox，避免把動物自己的移動誤認為鏡頭移動。
    """
    if prev_gray is None:
        return 0.0, 0.0

    h, w = prev_gray.shape[:2]

    mask = np.ones((h, w), dtype=np.uint8) * 255

    for bbox in exclude_boxes:
        if bbox is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 0

    corners = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=150, qualityLevel=0.01,
        minDistance=8, mask=mask
    )

    if corners is None or len(corners) < 8:
        return 0.0, 0.0

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, corners, None)

    if next_pts is None:
        return 0.0, 0.0

    status = status.reshape(-1)
    good_prev = corners[status == 1].reshape(-1, 2)
    good_next = next_pts[status == 1].reshape(-1, 2)

    if len(good_prev) < 8:
        return 0.0, 0.0

    diffs = good_next - good_prev

    # 用中位數而非平均，避免少數移動中的前景點干擾估計
    dx = float(np.median(diffs[:, 0]))
    dy = float(np.median(diffs[:, 1]))

    return dx, dy


def get_camera_offset(frame_a, frame_b, camera_motion_cache):
    """
    累加 frame_a -> frame_b 之間，每一幀對應的鏡頭位移，得到總位移量。
    """
    if frame_b <= frame_a:
        return 0.0, 0.0

    dx_total, dy_total = 0.0, 0.0

    for f in range(frame_a + 1, frame_b + 1):
        dx, dy = camera_motion_cache.get(f, (0.0, 0.0))
        dx_total += dx
        dy_total += dy

    return dx_total, dy_total


def median_filtered_mean(values):
    """
    先濾掉明顯離群值（超過中位數 3 倍的離群速度，通常來自單幀關鍵點錯誤跳動），
    再取平均，比直接取 mean 更抗雜訊。
    """
    if not values:
        return 0.0

    med = float(np.median(values))

    if med < 1e-6:
        filtered = values
    else:
        filtered = [v for v in values if v <= med * 3.0]

    if not filtered:
        filtered = values

    return sum(filtered) / len(filtered)


# ============================================================
# 行為辨識（改進版）
# 改進重點：
#   1. 用中位數 + 離群值過濾計算速度，避免單幀 keypoint 錯誤造成誤判
#   2. 加入「路徑效率」(淨位移/總路徑長)，區分「原地抖動」與「真的在移動」
#   3. 扣除鏡頭晃動造成的位移，避免手震/鏡頭移動被誤判成寵物在跑
# ============================================================

def classify_action(history, frame_w, frame_h, kpts, scores, camera_motion_cache=None):
    if len(history) < 8:
        return "analyzing"

    neck, tail, body_len = get_body_axis(kpts, scores)

    if body_len is None or body_len < 1:
        for h in reversed(list(history)):
            if h.get("body_len") and h["body_len"] > 1:
                body_len = h["body_len"]
                break

    if body_len is None or body_len < 1:
        return "unknown"

    hist_list = list(history)
    centers = [h["center"] for h in hist_list if h.get("center") is not None]

    if len(centers) < 3:
        return "unknown"

    # --- 計算(已扣除鏡頭晃動的)逐幀移動速度 ---
    speeds = []

    for i in range(1, len(hist_list)):
        prev_h = hist_list[i - 1]
        curr_h = hist_list[i]

        if prev_h.get("center") is None or curr_h.get("center") is None:
            continue

        dx = curr_h["center"][0] - prev_h["center"][0]
        dy = curr_h["center"][1] - prev_h["center"][1]

        if camera_motion_cache is not None and "frame_id" in prev_h and "frame_id" in curr_h:
            cam_dx, cam_dy = get_camera_offset(prev_h["frame_id"], curr_h["frame_id"], camera_motion_cache)
            dx -= cam_dx
            dy -= cam_dy

        d = math.hypot(dx, dy)
        speeds.append(d / body_len)

    avg_speed = median_filtered_mean(speeds)

    # --- 路徑效率：淨位移 / 總路徑長，用來分辨「真的移動」還是「原地抖動」 ---
    net_disp = distance(centers[0], centers[-1]) or 0.0
    total_path = sum(
        distance(centers[i - 1], centers[i]) or 0.0
        for i in range(1, len(centers))
    )
    path_efficiency = net_disp / total_path if total_path > 1e-6 else 0.0

    # --- 前腳移動 ---
    front_paws = [h["front_paw"] for h in hist_list if h.get("front_paw") is not None]

    front_speeds = []

    for i in range(1, len(front_paws)):
        d = distance(front_paws[i - 1], front_paws[i])

        if d is not None:
            front_speeds.append(d / body_len)

    avg_front_speed = median_filtered_mean(front_speeds)

    # 門檻
    STILL_THRES = 0.012
    WALK_THRES = 0.05
    PAW_MOVE_THRES = 0.02
    STILL_PAW_THRES = 0.03

    # 如果速度看起來很快，但路徑效率很低（原地晃動、非真正移動），視為 standing
    if avg_speed >= STILL_THRES and path_efficiency < PATH_EFFICIENCY_THRES and total_path > 0:
        features = get_smoothed_posture_features(hist_list, kpts, scores, body_len)
        if features:
            return classify_static_posture(features)
        return "standing"

    # running
    if avg_speed >= WALK_THRES:
        return "running"

    # walking
    if avg_speed >= STILL_THRES:
        if avg_front_speed > PAW_MOVE_THRES:
            return "walking"

        if avg_speed > STILL_THRES * 1.5:
            return "walking"

        return "standing"

    # 靜止時確認腳也沒動。SuperAnimal 的前腳關鍵點本身就會抖（靜止的貓實測中位數約 0.011），
    # 門檻要明顯高於這個雜訊，否則坐著不動也會被當成「腳在動」而直接判 standing
    if avg_speed < STILL_THRES and avg_front_speed < STILL_PAW_THRES:
        features = get_smoothed_posture_features(hist_list, kpts, scores, body_len)

        if not features:
            return "standing"

        return classify_static_posture(features)

    return "standing"


# ============================================================
# 骨架繪製
# ============================================================

def draw_pose(frame, kpts, scores, color=(255, 0, 0)):
    points = []

    for i in range(len(KEYPOINTS)):
        p = get_point(kpts, scores, i)
        points.append(p)

    for a, b in SKELETON_LINES:
        if a < len(points) and b < len(points):
            p1 = points[a]
            p2 = points[b]

            if p1 is not None and p2 is not None:
                cv2.line(frame, p1, p2, color, 2)

    for i, p in enumerate(points):
        if p is not None:
            cv2.circle(frame, p, 4, color, -1)

            cv2.putText(
                frame,
                str(i),
                (p[0] + 4, p[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )

    return points


CLIP_DURATION_SECONDS = float(os.getenv("CLIP_DURATION_SECONDS", "12"))


def make_action_video_live_camera_like(capture_source, mode):
    """
    即時來源行為分析（DeepLabCut 版）。

    跟舊版 MMPose 逐幀即時推論不一樣：SuperAnimal-Quadruped 沒有官方支援、
    穩定的單幀即時推論 API，只提供「丟一支影片檔」的批次分析介面。所以這裡
    改成：background 執行緒持續錄影，每 CLIP_DURATION_SECONDS 秒切一段，
    交給另一個執行緒依序拿去跑 DeepLabCut 分析、更新行為結果。

    即時影像（/video_feed）完全不受影響，每一幀進來就立刻更新；
    行為分析（/current）則是「N 秒前發生的事」，這個延遲是有意的設計，
    拿即時性換取穩定、官方支援的分析路徑。
    """
    print("========================================")
    print("Step 2: 即時來源行為分析（DeepLabCut SuperAnimal-Quadruped）")
    print("========================================")

    cap = open_capture(capture_source, mode)
    if not cap.isOpened():
        raise RuntimeError("無法開啟即時影像來源")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 15.0

    ret, frame = cap.read()
    if not ret or frame is None:
        raise RuntimeError("無法從即時來源讀到第一幀")

    frame_h, frame_w = frame.shape[:2]
    print("即時來源解析度：", frame_w, "x", frame_h)
    print("FPS：", fps)
    print("每段分析長度：", CLIP_DURATION_SECONDS, "秒")

    stop_event = threading.Event()
    clip_queue = queue.Queue(maxsize=2)  # 分析跟不上時，寧可丟掉舊片段也不要延遲越積越大
    frames_per_clip = max(1, int(round(fps * CLIP_DURATION_SECONDS)))

    def capture_and_clip_worker():
        nonlocal frame

        clip_index = 0
        writer = None
        frame_count_in_clip = 0

        try:
            while not stop_event.is_set():
                if frame is None:
                    break

                update_latest_frame(frame)

                if writer is None:
                    clip_path = OUTPUT_DIR / f"_live_clip_{clip_index}.mp4"
                    writer = cv2.VideoWriter(
                        str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h)
                    )

                writer.write(frame)
                frame_count_in_clip += 1

                if frame_count_in_clip >= frames_per_clip:
                    writer.release()
                    writer = None
                    frame_count_in_clip = 0
                    clip_index += 1

                    try:
                        clip_queue.put_nowait(clip_path)
                    except queue.Full:
                        print("⚠️ DeepLabCut 分析跟不上，捨棄這一段影片避免延遲累積")
                        clip_path.unlink(missing_ok=True)

                ret_local, frame = cap.read()
                if not ret_local:
                    frame = None
        finally:
            if writer is not None:
                writer.release()
            stop_event.set()
            clip_queue.put(None)  # 通知分析執行緒可以結束了

    capture_thread = threading.Thread(target=capture_and_clip_worker, daemon=True)
    capture_thread.start()

    tracks = {}
    behavior_segments = {}
    active_segments = {}
    camera_motion_cache = {}
    prev_gray = None
    frame_id = 0

    try:
        while True:
            clip_path = clip_queue.get()

            if clip_path is None:
                break

            try:
                frames_instances = run_dlc_clip_inference(clip_path)
            except Exception as e:
                print("❌ DeepLabCut 分析這段影片失敗：", e)
                frames_instances = []

            # 讀回這段影片的真實畫面，顏色直方圖／鏡頭晃動補償都需要真的像素，不能用黑畫面代替
            clip_cap = cv2.VideoCapture(str(clip_path))

            for instances in frames_instances:
                ret_clip, clip_frame = clip_cap.read()
                if not ret_clip or clip_frame is None:
                    clip_frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

                _, _, prev_gray = process_behavior_frame(
                    clip_frame,
                    frame_id,
                    instances,
                    tracks,
                    behavior_segments,
                    active_segments,
                    frame_w,
                    frame_h,
                    camera_motion_cache,
                    prev_gray,
                )

                frame_id += 1

            clip_cap.release()
            clip_path.unlink(missing_ok=True)

            update_latest_result(tracks)

            print(
                f"分析完一段影片，累積到第 {frame_id} 幀，"
                f"目前 tracks 數量 {len(tracks)}"
            )
    finally:
        stop_event.set()
        capture_thread.join(timeout=5.0)
        cap.release()

    print("即時來源行為分析結束")

    write_behavior_log(
        behavior_segments,
        active_segments,
        fps,
        runtime_source_label,
        BEHAVIOR_LOG_PATH
    )


def run_batch_video_analysis(video_path):
    """
    離線批次分析（測試用）：整支影片檔一次跑完 DeepLabCut，逐幀畫出骨架 +
    行為標籤，輸出成標註影片（output/{video_name}_行為辨識影片.mp4）+ 行為紀錄文字檔。

    跟即時的 make_action_video_live_camera_like 不同，這裡不用 clip 佇列切段，
    整支影片只送進 DeepLabCut 跑一次，不會因為分析速度跟不上讀檔速度而丟幀，
    適合離線驗證「這支影片的行為辨識結果」，不追求即時性。
    """
    video_path = Path(video_path)

    print("========================================")
    print("批次影片行為分析（DeepLabCut SuperAnimal-Quadruped）")
    print("========================================")
    print("影片：", video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("無法開啟影片")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print("解析度：", frame_w, "x", frame_h)
    print("FPS：", fps)

    print("Step 1: 執行 DeepLabCut 姿態估計（整支影片）...")
    frames_instances = run_dlc_clip_inference(video_path)
    print(f"共取得 {len(frames_instances)} 幀的姿態結果")

    result_path = OUTPUT_DIR / f"{runtime_source_label}_行為辨識影片.mp4"
    # OpenCV 在 Windows 上寫中文路徑常會失敗，先寫成英文暫存檔，寫完再改成中文檔名
    tmp_result_path = OUTPUT_DIR / "_writing_result.mp4"
    out = cv2.VideoWriter(
        str(tmp_result_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h)
    )

    tracks = {}
    behavior_segments = {}
    active_segments = {}
    camera_motion_cache = {}
    prev_gray = None
    frame_id = 0

    read_cap = cv2.VideoCapture(str(video_path))

    print("Step 2: 逐幀追蹤 + 行為分類 + 畫框...")

    for instances in frames_instances:
        ret, frame = read_cap.read()
        if not ret or frame is None:
            break

        frame_actions, _, prev_gray = process_behavior_frame(
            frame,
            frame_id,
            instances,
            tracks,
            behavior_segments,
            active_segments,
            frame_w,
            frame_h,
            camera_motion_cache,
            prev_gray,
        )

        cv2.putText(frame, "Pet Behavior", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4)
        for line_idx, (pet_id, action, color) in enumerate(frame_actions):
            cv2.putText(
                frame,
                f"Pet {pet_id}: {action}",
                (30, 140 + line_idx * 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.8,
                color,
                4,
            )

        out.write(frame)
        frame_id += 1

        if frame_id % 100 == 0:
            print(f"已處理 {frame_id}/{len(frames_instances)} 幀")

    read_cap.release()
    out.release()

    os.replace(tmp_result_path, result_path)

    write_behavior_log(
        behavior_segments,
        active_segments,
        fps,
        runtime_source_label,
        BEHAVIOR_LOG_PATH,
    )

    print("✅ 批次分析完成")
    print("標註影片：", result_path)


def main():
    print("========================================")
    print("一鍵寵物骨架 + 行為辨識流程開始")
    print("========================================")

    mode, capture_source, source_label = resolve_input_source()
    configure_runtime_paths(source_label)

    print("來源模式：", mode)
    print("來源標籤：", runtime_source_label)
    print("輸出資料夾：", OUTPUT_DIR)

    if mode == "video":
        run_batch_video_analysis(capture_source)
    else:
        make_action_video_live_camera_like(capture_source, mode)

    print("========================================")
    print("全部完成")
    print("行為紀錄：", BEHAVIOR_LOG_PATH)
    print("========================================")


if __name__ == "__main__":
    main()
