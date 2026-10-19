"""
追蹤 / 身分穩定度評估工具

對一支「有 ground truth 的多隻寵物影片」（目前由 make_composite.py 產生），
跑現在 main_model.py 裡真正在用的偵測 + 追蹤 + 行為判斷流程，
再跟 ground truth 比對，量化：
  - DeepLabCut 有沒有漏偵測（這不是追蹤器的錯，要分開看）
  - 追蹤器有沒有把同一隻拆成多個編號（fragmentation）、編號有沒有中途換人（ID switch）
  - 整體身分正確率（IDF1）
  - 行為標籤會不會在短時間內來回跳（直接影響活動摘要的準確度）

設計重點：
  1. 本工具只「讀」main_model 的追蹤狀態，不修改 main_model.py，
     所以改追蹤器（步驟 2）之後，直接重跑就能比較前後差異。
  2. DeepLabCut 的結果會快取，改追蹤器時不用每次重跑幾分鐘的姿態估計
     （影片內容改變會自動重算；也可以用 --refresh 強制重算）。
  3. 指標計算是純函式，--selftest 用合成資料驗證指標本身算得對。

用法（在專案根目錄）：
  venv\\Scripts\\python.exe evaluation\\evaluate_tracking.py --selftest
  venv\\Scripts\\python.exe evaluation\\evaluate_tracking.py input\\composite_side_by_side.mp4
  venv\\Scripts\\python.exe evaluation\\evaluate_tracking.py input\\composite_swap.mp4 --debug-video --tag baseline
"""

import argparse
import ast
import inspect
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_DIR / "output" / "eval"
CACHE_DIR = EVAL_DIR / "cache"

# 判讀用的門檻（只是參考，不影響指標數值）
GOOD_IDF1 = 0.95
SHORT_SEGMENT_SECONDS = 1.0          # 短於這個秒數的行為區段視為「閃一下」
MIN_OBS_FOR_SEGMENT_MAJORITY = 10    # 某編號在某段至少要有幾筆觀測，才判斷它在那一段「是誰」

# 記錄在報告裡，方便之後比較「改了哪些參數」
TRACKER_PARAMS = [
    "MAX_MISSING_FRAMES", "MATCH_THRES", "MIN_VALID_KEYPOINTS", "DUPLICATE_IOU_THRES",
    "TRACK_MIN_HITS", "TENTATIVE_MAX_MISSING", "COLOR_HIST_BINS",
    "W_CENTER", "W_BODYSIZE", "W_FEATURE", "W_COLOR",
    "REFERENCE_FPS", "ACTION_VOTE_SECONDS", "HISTORY_SECONDS", "MIN_HISTORY_SECONDS", "MIN_ACTION_SECONDS",
]

# 這些「標籤」不是真正的行為，算行為區段 / 閃動時要排除
NON_BEHAVIOR_LABELS = (None, "analyzing")


# ============================================================
# Ground truth
# ============================================================

def load_gt(video_path, gt_arg):
    gt_path = Path(gt_arg) if gt_arg else Path(video_path).with_suffix(".gt.json")
    if not gt_path.exists():
        sys.exit(f"找不到 ground truth：{gt_path}（請先用 make_composite.py 產生）")
    return json.loads(gt_path.read_text(encoding="utf-8"))


def segment_index(gt, frame_idx):
    for i, seg in enumerate(gt["segments"]):
        if seg["start_frame"] <= frame_idx <= seg["end_frame"]:
            return i
    return None


def true_identity(gt, frame_idx, x):
    """依畫面位置（左 / 右面板）與時間（對調前 / 後）決定這個點「真正」是哪一隻。"""
    i = segment_index(gt, frame_idx)
    if i is None:
        return "?"
    seg = gt["segments"][i]
    return seg["left"] if x < gt["panel_width"] else seg["right"]


# ============================================================
# 跑偵測 + 追蹤（會用到 main_model，所以延後 import，--selftest 不需要載入 deeplabcut）
# ============================================================

def load_main_model():
    sys.path.insert(0, str(REPO_DIR))
    import main_model
    return main_model


def get_instances(main_model, video_path, refresh):
    st = video_path.stat()
    cache = CACHE_DIR / f"{video_path.stem}_{st.st_size}_{int(st.st_mtime)}.json"

    if cache.exists() and not refresh:
        print(f"使用快取的 DeepLabCut 結果：{cache.name}")
        return json.loads(cache.read_text(encoding="utf-8"))

    print("執行 DeepLabCut 姿態估計（第一次較慢，結果會快取）...")
    t0 = time.time()
    frames = main_model.run_dlc_clip_inference(video_path)
    print(f"DeepLabCut 完成：{len(frames)} 幀，用時 {time.time() - t0:.0f} 秒")

    if not frames:
        sys.exit("DeepLabCut 沒有產生任何結果")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(frames), encoding="utf-8")
    return frames


def draw_debug(frame, gt, frame_idx, obs, detections=None):
    h = frame.shape[0]
    scale = h / 720.0
    unlabeled = gt.get("mode") == "single"

    if not unlabeled:
        cv2.line(frame, (gt["panel_width"], 0), (gt["panel_width"], h), (255, 255, 255), 1)

    # 無標註模式：把偵測器給的每個框（過濾後）用細白線畫出來，才看得出「有框但沒被追蹤」或「多框」
    for d in detections or []:
        if d.get("bbox") is not None:
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    for o in obs:
        # 無標註模式用「追蹤實例編號 U」：Pet 編號會被重複使用，不能拿來判斷是不是同一條追蹤
        text = f"U{o['uid']}" if unlabeled else f"T{o['pid']}={o['true']}"
        pos = (int(o["x"]) - 40, int(o["y"]) + int(60 * scale))
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, (0, 0, 0), 4)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, (255, 255, 255), 2)

    cv2.putText(frame, f"frame {frame_idx} seg {segment_index(gt, frame_idx)}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (255, 255, 255), 1)


def run_tracker(main_model, video_path, frames_instances, gt, debug_path=None, max_frames=None, stride=1):
    """
    重現 main_model.run_batch_video_analysis() 的逐幀流程，
    每幀結束後讀出「目前已確認、且這一幀有被配對到」的追蹤，當作觀測紀錄。

    stride > 1：只處理每第 stride 幀（模擬「不用分析每一幀」的抽樣），追蹤器看到的 fps 與幀編號
    都是抽樣後的（fps / stride），回傳的 fps 也是抽樣後的，之後的指標才會以正確的秒數計算。
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = (cap.get(cv2.CAP_PROP_FPS) or gt.get("fps", 30.0)) / stride
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if debug_path:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(debug_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h))

    tracks, behavior_segments, active_segments, camera_motion_cache = {}, {}, {}, {}
    prev_gray = None
    records = []

    # 追蹤「實例」的生滅：Pet 編號會被重複使用，所以用 track dict 物件本身當身分。
    # keepalive 把看過的 dict 都留著，避免被回收後 id() 被新物件重用而認錯
    uids, instances, keepalive = {}, {}, []

    n = len(frames_instances) if max_frames is None else min(max_frames, len(frames_instances))

    # 新版 main_model 會依 fps 換算時間視窗；舊版沒有這個參數，也要能跑（才能做前後比較）
    extra = {"fps": fps} if "fps" in inspect.signature(main_model.process_behavior_frame).parameters else {}

    for source_idx in range(n):
        if source_idx % stride:
            if not cap.grab():
                break
            continue

        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_idx = source_idx // stride  # 抽樣後的幀編號（stride=1 時就是原本的幀編號）

        _, detections, prev_gray = main_model.process_behavior_frame(
            frame, frame_idx, frames_instances[source_idx],
            tracks, behavior_segments, active_segments,
            frame_w, frame_h, camera_motion_cache, prev_gray,
            **extra,
        )

        det_true = [true_identity(gt, frame_idx, d["center"][0]) for d in detections]

        for pid, t in tracks.items():
            uid = uids.get(id(t))
            if uid is None:
                uid = len(uids) + 1
                uids[id(t)] = uid
                keepalive.append(t)
                instances[uid] = {"pid": pid, "born": frame_idx, "last_seen": frame_idx,
                                  "seen_frames": 0, "confirmed_at": None}
            inst = instances[uid]
            if t["missing_frames"] == 0:
                inst["last_seen"] = frame_idx
                inst["seen_frames"] += 1
            if t["confirmed"] and inst["confirmed_at"] is None:
                inst["confirmed_at"] = frame_idx

        obs = []
        for pid, t in tracks.items():
            if t["missing_frames"] != 0 or not t["confirmed"]:
                continue  # 這一幀沒被配對到，或還沒確認，不算「有輸出結果」

            cx, cy = t["last_center"]
            history = t.get("action_history")
            raw_action = history[-1] if history else None

            # 「最終輸出的標籤」：新版 main_model 會存在 current_action；舊版是最近幾幀的多數決
            action = t.get("current_action")
            if action is None and history:
                action = Counter(history).most_common(1)[0][0]

            obs.append({
                "pid": pid, "uid": uids[id(t)], "true": true_identity(gt, frame_idx, cx), "x": cx, "y": cy,
                "action": action, "raw_action": raw_action,
            })

        records.append({"frame": frame_idx, "det_true": det_true, "n_det": len(detections), "obs": obs})

        if writer is not None:
            draw_debug(frame, gt, frame_idx, obs, detections)
            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()

    # 實際寫進行為紀錄（txt / 之後上傳）的區段；還在進行中的區段也收進來，跟 write_behavior_log 一樣
    log_segments = {}
    for pid in set(behavior_segments) | set(active_segments):
        segs = [dict(s) for s in behavior_segments.get(pid, [])]
        if pid in active_segments:
            segs.append(dict(active_segments[pid]))
        log_segments[pid] = segs

    return records, fps, log_segments, instances


# ============================================================
# 指標計算（純函式，不依賴 DeepLabCut）
# ============================================================

def pick_observation(candidates, prev_pid):
    """
    同一個身分區域內同時有多個已確認追蹤（＝同一隻被重複追蹤）時，
    優先沿用上一幀選到的那個，避免重複追蹤被誤算成 ID 切換。
    """
    for o in candidates:
        if o["pid"] == prev_pid:
            return o
    return min(candidates, key=lambda o: o["pid"])


def behavior_runs(labeled):
    """把 [(frame, action)] 依「連續幀 + 相同行為」切成區段。"""
    runs = []
    for frame, action in labeled:
        if runs and runs[-1]["action"] == action and frame == runs[-1]["end"] + 1:
            runs[-1]["end"] = frame
        else:
            runs.append({"action": action, "start": frame, "end": frame})
    return runs


def behavior_stats(labeled, fps):
    """一串逐幀行為標籤的區段數、短區段數、每分鐘切換次數、各行為秒數。"""
    labeled = [(f, a) for f, a in labeled if a not in NON_BEHAVIOR_LABELS]
    runs = behavior_runs(labeled)
    durations = [(r["end"] - r["start"] + 1) / fps for r in runs]

    distribution = Counter()
    for r in runs:
        distribution[r["action"]] += (r["end"] - r["start"] + 1) / fps

    observed_seconds = len(labeled) / fps
    return {
        "segments": len(runs),
        "short_segments": sum(1 for d in durations if d < SHORT_SEGMENT_SECONDS),
        "changes_per_minute": (len(runs) - 1) / (observed_seconds / 60) if observed_seconds > 0 and len(runs) > 1 else 0.0,
        "seconds": {k: round(v, 2) for k, v in sorted(distribution.items(), key=lambda kv: -kv[1])},
    }


def summarize_log(log_segments, fps):
    """實際輸出的行為區段（依追蹤編號）：區段數、短區段數、各行為秒數。"""
    out = {}
    for pid, segs in sorted(log_segments.items()):
        real = [s for s in segs if s["action"] not in NON_BEHAVIOR_LABELS]
        durations = [(s["end_frame"] - s["start_frame"] + 1) / fps for s in real]

        distribution = Counter()
        for s, d in zip(real, durations):
            distribution[s["action"]] += d

        out[pid] = {
            "segments": len(real),
            "short_segments": sum(1 for d in durations if d < SHORT_SEGMENT_SECONDS),
            "seconds": {k: round(v, 2) for k, v in sorted(distribution.items(), key=lambda kv: -kv[1])},
            "timeline": [
                {"action": s["action"], "start": round(s["start_frame"] / fps, 1), "end": round((s["end_frame"] + 1) / fps, 1)}
                for s in real
            ],
        }
    return out


def clock(seconds):
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


def compute_metrics(records, gt, fps):
    ids = sorted(gt["identities"])
    n_frames = len(records)

    det_frames = dict.fromkeys(ids, 0)
    det_dup_frames = dict.fromkeys(ids, 0)
    track_frames = dict.fromkeys(ids, 0)
    track_dup_frames = dict.fromkeys(ids, 0)

    counts = defaultdict(Counter)                         # counts[pid][true] = 觀測筆數
    seg_obs = defaultdict(list)                           # seg_obs[段落] = [(pid, true)]
    pid_seg_true = defaultdict(lambda: defaultdict(Counter))
    chosen = {i: [] for i in ids}                         # 每個身分每幀「代表的追蹤」：(frame, pid, action, raw_action)
    prev_pid = dict.fromkeys(ids)
    track_span = {}

    for rec in records:
        f = rec["frame"]
        seg = segment_index(gt, f)

        dt = Counter(rec["det_true"])
        for i in ids:
            if dt[i] >= 1:
                det_frames[i] += 1
            if dt[i] > 1:
                det_dup_frames[i] += 1

        by_true = defaultdict(list)
        for o in rec["obs"]:
            counts[o["pid"]][o["true"]] += 1
            seg_obs[seg].append((o["pid"], o["true"]))
            pid_seg_true[o["pid"]][seg][o["true"]] += 1
            by_true[o["true"]].append(o)
            span = track_span.setdefault(o["pid"], [f, f])
            span[1] = f

        for i in ids:
            cand = by_true.get(i, [])
            if cand:
                track_frames[i] += 1
                o = pick_observation(cand, prev_pid[i])
                prev_pid[i] = o["pid"]
                chosen[i].append((f, o["pid"], o["action"], o.get("raw_action", o["action"])))
            if len(cand) > 1:
                track_dup_frames[i] += 1

    # --- 整體身分正確率（IDF1）：編號與真實身分做一對一最佳配對 ---
    pids = sorted(counts)
    assign = {}
    idtp = 0

    if pids:
        matrix = np.zeros((len(pids), len(ids)))
        for r, pid in enumerate(pids):
            for c, i in enumerate(ids):
                matrix[r, c] = counts[pid][i]

        rows, cols = linear_sum_assignment(-matrix)
        for r, c in zip(rows, cols):
            if matrix[r, c] > 0:
                assign[pids[r]] = ids[c]
                idtp += int(matrix[r, c])

    total_obs = sum(sum(c.values()) for c in counts.values())
    gt_slots = n_frames * len(ids)  # ground truth 裡每個身分每一幀都在畫面中
    idp = idtp / total_obs if total_obs else 0.0
    idr = idtp / gt_slots if gt_slots else 0.0
    idf1 = 2 * idtp / (total_obs + gt_slots) if (total_obs + gt_slots) else 0.0

    # --- 每個身分的指標 ---
    per_identity = {}
    for i in ids:
        seq = chosen[i]
        switch_frames = [seq[k][0] for k in range(1, len(seq)) if seq[k][1] != seq[k - 1][1]]
        out_stats = behavior_stats([(f, a) for f, _, a, _ in seq], fps)
        raw_stats = behavior_stats([(f, r) for f, _, _, r in seq], fps)

        per_identity[i] = {
            "detected_frame_ratio": det_frames[i] / n_frames if n_frames else 0.0,
            "detection_duplicate_frames": det_dup_frames[i],
            "tracked_frame_ratio": track_frames[i] / n_frames if n_frames else 0.0,
            "track_duplicate_frames": track_dup_frames[i],
            "track_ids_used": sorted({p for _, p, _, _ in seq}),
            "id_switches": len(switch_frames),
            "id_switch_frames": switch_frames[:20],
            # 最終輸出（會進行為紀錄 / 活動摘要的標籤）
            "behavior_segments": out_stats["segments"],
            "short_behavior_segments": out_stats["short_segments"],
            "behavior_changes_per_minute": out_stats["changes_per_minute"],
            "behavior_seconds": out_stats["seconds"],
            # 原始逐幀判斷（還沒平滑），用來看平滑層有沒有效
            "raw_behavior_segments": raw_stats["segments"],
            "raw_short_behavior_segments": raw_stats["short_segments"],
        }

    # --- 每個追蹤編號的組成 ---
    tracks_info = {}
    for pid in pids:
        c = counts[pid]
        total = sum(c.values())
        major, major_n = c.most_common(1)[0]
        tracks_info[pid] = {
            "observations": total,
            "first_frame": track_span[pid][0],
            "last_frame": track_span[pid][1],
            "majority_identity": major,
            "purity": major_n / total,
            "assigned_identity": assign.get(pid),
        }

    # --- 分段準確度（swap 模式用）：沿用整體最佳配對，看對調前 / 後各自準不準 ---
    segment_accuracy = {}
    for seg, lst in seg_obs.items():
        if seg is None:
            continue
        ok = sum(1 for pid, true in lst if assign.get(pid) == true)
        segment_accuracy[str(seg)] = {"observations": len(lst), "accuracy": ok / len(lst) if lst else 0.0}

    # --- 編號是否「中途換了身分」：同一個編號在不同段落多數都是不同的真實身分 ---
    identity_changed_tracks = []
    for pid in pids:
        majors = {}
        for seg, c in pid_seg_true[pid].items():
            if seg is not None and sum(c.values()) >= MIN_OBS_FOR_SEGMENT_MAJORITY:
                majors[seg] = c.most_common(1)[0][0]
        if len(set(majors.values())) > 1:
            identity_changed_tracks.append({
                "pid": pid,
                "identity_by_segment": {str(s): v for s, v in sorted(majors.items())},
            })

    return {
        "frames": n_frames,
        "identities": ids,
        "idf1": idf1,
        "idp": idp,
        "idr": idr,
        "total_observations": total_obs,
        "per_identity": per_identity,
        "tracks": tracks_info,
        "segment_accuracy": segment_accuracy,
        "identity_changed_tracks": identity_changed_tracks,
    }


# ============================================================
# 無標註的多隻模式（沒有 ground truth：真實的多隻影片）
# 不量「是不是對的那一隻」，只量不需要標準答案、但一看就知道有沒有問題的東西：
# 偵測 / 追蹤到幾隻、總共生成了幾條追蹤（跟實際隻數比）、碎片、編號重複使用、行為閃動
# ============================================================

FRAGMENT_SECONDS = 2.0   # 確認後壽命短於這個秒數的追蹤，視為碎片 / 暫時性的誤偵測


def runs_ignoring_gaps(labeled):
    """[(frame, action)] → 只在「行為改變」時才切區段（被遮住幾幀不算斷開）。"""
    runs = []
    for frame, action in labeled:
        if runs and runs[-1]["action"] == action:
            runs[-1]["end"] = frame
        else:
            runs.append({"action": action, "start": frame, "end": frame})
    return runs


def compute_unlabeled_metrics(records, instances, fps, expected_count=None):
    n = len(records)
    det_hist = Counter(r["n_det"] for r in records)
    obs_hist = Counter(len(r["obs"]) for r in records)

    confirmed = {u: i for u, i in instances.items() if i["confirmed_at"] is not None}

    labeled = defaultdict(list)
    for rec in records:
        for o in rec["obs"]:
            labeled[o["uid"]].append((rec["frame"], o["action"]))

    tracks = []
    total_segments = total_short = 0
    for uid, inst in sorted(confirmed.items()):
        runs = runs_ignoring_gaps([(f, a) for f, a in labeled[uid] if a not in NON_BEHAVIOR_LABELS])
        durations = [(r["end"] - r["start"] + 1) / fps for r in runs]
        short = sum(1 for d in durations if d < SHORT_SEGMENT_SECONDS)
        total_segments += len(runs)
        total_short += short

        lifespan = inst["last_seen"] - inst["born"] + 1
        tracks.append({
            "uid": uid,
            "pid": inst["pid"],
            "born": inst["born"],
            "confirmed_at": inst["confirmed_at"],
            "last_seen": inst["last_seen"],
            "seconds": round((inst["last_seen"] - inst["confirmed_at"] + 1) / fps, 1),
            "seen_frames": inst["seen_frames"],
            "coverage": inst["seen_frames"] / lifespan if lifespan else 0.0,
            "behavior_segments": len(runs),
            "short_behavior_segments": short,
        })

    fragments = [t["uid"] for t in tracks if t["seconds"] < FRAGMENT_SECONDS]
    pid_instances = Counter(i["pid"] for i in instances.values())

    m = {
        "frames": n,
        "seconds": n / fps if fps else 0.0,
        "detections_per_frame": {str(k): v / n for k, v in sorted(det_hist.items())} if n else {},
        "detections_mean": sum(k * v for k, v in det_hist.items()) / n if n else 0.0,
        "detections_max": max(det_hist) if det_hist else 0,
        "confirmed_per_frame": {str(k): v / n for k, v in sorted(obs_hist.items())} if n else {},
        "confirmed_mean": sum(k * v for k, v in obs_hist.items()) / n if n else 0.0,
        "confirmed_max": max(obs_hist) if obs_hist else 0,
        "tracks_created": len(instances),
        "tracks_confirmed": len(confirmed),
        "tracks_never_confirmed": len(instances) - len(confirmed),
        "fragment_tracks": fragments,
        "late_confirmed_tracks": sum(1 for t in tracks if t["confirmed_at"] > fps),
        "reused_pids": {str(p): c for p, c in sorted(pid_instances.items()) if c > 1},
        "tracks": tracks,
        "behavior_segments": total_segments,
        "short_behavior_segments": total_short,
        "expected_count": expected_count,
    }

    if expected_count is not None and n:
        m["frames_with_expected_count"] = obs_hist.get(expected_count, 0) / n
        m["frames_under_count"] = sum(v for k, v in obs_hist.items() if k < expected_count) / n
        m["frames_over_count"] = sum(v for k, v in obs_hist.items() if k > expected_count) / n
        m["extra_tracks"] = len(confirmed) - expected_count

    return m


def percent_line(hist):
    return "｜".join(f"{k} 隻 {v:.0%}" for k, v in hist.items() if v >= 0.005) or "（無）"


def build_unlabeled_verdict(m):
    notes = []
    expected = m["expected_count"]

    if expected is None:
        notes.append(f"ℹ 沒有提供實際隻數（--expected-count），所以下面只能列數字，無法判斷「多出幾條追蹤」")
    else:
        if m["extra_tracks"] > 0:
            notes.append(
                f"⚠ 共生成 {m['tracks_confirmed']} 條已確認追蹤，但實際只有 {expected} 隻 → 多出 {m['extra_tracks']} 條"
                f"（同一隻被拆成多條 / 換了新編號，或把玩具、人腳當成寵物）"
            )
        if m["detections_mean"] < expected * 0.8:
            notes.append(
                f"ℹ DeepLabCut 平均每幀只偵測到 {m['detections_mean']:.1f} 隻（實際 {expected} 隻）"
                f"——漏偵測，這部分不是追蹤器的問題"
            )
        notes.append(
            f"ℹ 追蹤輸出剛好 {expected} 隻的幀只有 {m['frames_with_expected_count']:.0%}"
            f"（少於 {m['frames_under_count']:.0%}、多於 {m['frames_over_count']:.0%}）"
        )

    if m["fragment_tracks"]:
        notes.append(f"⚠ {len(m['fragment_tracks'])} 條追蹤壽命短於 {FRAGMENT_SECONDS:g} 秒（碎片）：U{m['fragment_tracks'][:8]}")
    if m["reused_pids"]:
        notes.append(f"ℹ 編號 {list(m['reused_pids'])} 被不同的追蹤重複使用——「Pet N」不能當成固定的身分")
    if m["tracks_never_confirmed"] >= 5:
        notes.append(f"ℹ 另外有 {m['tracks_never_confirmed']} 條暫時追蹤從未被確認（一閃而過的偵測雜訊，已被過濾，不影響輸出）")
    if m["short_behavior_segments"] >= 3:
        notes.append(
            f"⚠ 行為區段共 {m['behavior_segments']} 段，其中 {m['short_behavior_segments']} 段短於 {SHORT_SEGMENT_SECONDS:g} 秒"
        )

    return notes


def format_unlabeled_report(meta, m):
    lines = []
    add = lines.append

    add("=" * 60)
    add("多隻影片評估（無 ground truth）")
    add("=" * 60)
    add(f"影片：{meta['video']}（{m['frames']} 幀，{meta['fps']:.2f} fps，{m['seconds']:.1f} 秒）")
    add(f"追蹤器版本：git {meta['git']}")
    if meta.get("overrides"):
        add(f"臨時覆寫的參數：{meta['overrides']}")
    if m["expected_count"] is not None:
        add(f"實際隻數（你提供的）：{m['expected_count']}")
    add(f"執行時間：{meta['time']}")
    add("")

    add("【DeepLabCut 偵測】每幀偵測到幾隻（過濾玩具 / 重複框之後）")
    add(f"  平均 {m['detections_mean']:.2f} 隻，最多 {m['detections_max']} 隻")
    add(f"  {percent_line(m['detections_per_frame'])}")
    add("")

    add("【追蹤輸出】每幀「已確認」的追蹤有幾條")
    add(f"  平均 {m['confirmed_mean']:.2f} 條，最多 {m['confirmed_max']} 條")
    add(f"  {percent_line(m['confirmed_per_frame'])}")
    add("")

    add("【追蹤的生滅】（越接近實際隻數越好；遠超過就是在拆碎 / 換編號）")
    add(f"  總共生成 {m['tracks_created']} 條追蹤，其中已確認 {m['tracks_confirmed']} 條、從未確認 {m['tracks_never_confirmed']} 條")
    add(f"  壽命短於 {FRAGMENT_SECONDS:g} 秒的碎片 {len(m['fragment_tracks'])} 條；影片 1 秒之後才新出現的已確認追蹤 {m['late_confirmed_tracks']} 條")
    if m["reused_pids"]:
        add(f"  被重複使用的 Pet 編號：{m['reused_pids']}（編號 → 被幾條不同追蹤用過）")
    add("")

    add("【已確認追蹤清單】U = 追蹤實例編號（debug 影片上標的就是這個）")
    for t in m["tracks"]:
        add(
            f"  U{t['uid']}（Pet {t['pid']}）：第 {t['confirmed_at']}~{t['last_seen']} 幀，{t['seconds']:g} 秒，"
            f"實際看到 {t['coverage']:.0%}，行為 {t['behavior_segments']} 段（短 {t['short_behavior_segments']}）"
        )
    add("")

    add("【判讀】（門檻僅供參考）")
    for note in build_unlabeled_verdict(m):
        add(f"  {note}")

    return "\n".join(lines)


# ============================================================
# 報告
# ============================================================

def build_verdict(m):
    notes = []

    if m["idf1"] < GOOD_IDF1:
        notes.append(f"⚠ 整體身分正確率 IDF1 = {m['idf1']:.2f}，低於參考門檻 {GOOD_IDF1}")

    for i, p in m["per_identity"].items():
        if p["detected_frame_ratio"] < 0.9:
            missed = 1 - p["detected_frame_ratio"]
            notes.append(f"ℹ 身分 {i} 有 {missed:.0%} 的幀 DeepLabCut 根本沒偵測到——這部分不是追蹤器的問題")
        if p["id_switches"]:
            notes.append(f"⚠ 身分 {i} 發生 {p['id_switches']} 次 ID 切換（幀 {p['id_switch_frames'][:5]}...）")
        if len(p["track_ids_used"]) > 1:
            notes.append(f"⚠ 身分 {i} 被拆成 {len(p['track_ids_used'])} 個編號 {p['track_ids_used']}（fragmentation）")
        if p["track_duplicate_frames"]:
            notes.append(f"⚠ 身分 {i} 有 {p['track_duplicate_frames']} 幀同時被 2 個以上的追蹤佔住（重複追蹤）")
        if p["short_behavior_segments"] >= 3:
            notes.append(
                f"⚠ 身分 {i} 的行為區段有 {p['short_behavior_segments']} 個短於 {SHORT_SEGMENT_SECONDS:g} 秒"
                f"（共 {p['behavior_segments']} 段）——行為標籤在閃動，會弄髒活動摘要"
            )

    if m["identity_changed_tracks"]:
        pids = [t["pid"] for t in m["identity_changed_tracks"]]
        notes.append(f"⚠ 編號 {pids} 在中途換了身分——編號跟著「畫面位置」走，而不是跟著「動物外觀」走")

    if not notes:
        notes.append("✅ 沒有發現問題（在這支影片上）")

    return notes


def format_report(meta, m):
    lines = []
    add = lines.append

    add("=" * 60)
    add("追蹤 / 身分穩定度評估")
    add("=" * 60)
    stride = meta.get("stride", 1)
    sampling = f"，抽樣：每 {stride} 幀取 1 幀" if stride > 1 else ""
    add(f"影片：{meta['video']}（{m['frames']} 幀，{meta['fps']:.1f} fps，模式 {meta['mode']}{sampling}）")
    add(f"追蹤器版本：git {meta['git']}")
    if meta.get("overrides"):
        add(f"臨時覆寫的參數：{meta['overrides']}")
    add(f"執行時間：{meta['time']}")
    add("")

    add("【DeepLabCut 偵測】每個身分「至少有一個偵測」的幀數比例")
    for i, p in m["per_identity"].items():
        dup = f"，同一隻被重複偵測 {p['detection_duplicate_frames']} 幀" if p["detection_duplicate_frames"] else ""
        add(f"  身分 {i}：{p['detected_frame_ratio']:.1%}{dup}")
    add("")

    add("【追蹤器輸出】每個身分「有已確認追蹤」的幀數比例")
    for i, p in m["per_identity"].items():
        add(f"  身分 {i}：{p['tracked_frame_ratio']:.1%}　使用的編號 {p['track_ids_used']}　ID 切換 {p['id_switches']} 次")
    add("")

    add("【身分正確率】編號與真實身分做一對一最佳配對")
    add(f"  IDF1 = {m['idf1']:.3f}　（精確率 {m['idp']:.3f}，召回率 {m['idr']:.3f}）")
    add("")

    add("【追蹤編號清單】")
    for pid, t in m["tracks"].items():
        add(
            f"  Pet {pid}：{t['observations']} 筆觀測（第 {t['first_frame']}~{t['last_frame']} 幀），"
            f"多數是身分 {t['majority_identity']}（純度 {t['purity']:.0%}），配對到 {t['assigned_identity']}"
        )
    add("")

    if len(m["segment_accuracy"]) > 1:
        add("【分段準確度】（沿用整體最佳配對；swap 模式下，第 1 段 = 對調前，第 2 段 = 對調後）")
        for seg, s in sorted(m["segment_accuracy"].items()):
            add(f"  第 {int(seg) + 1} 段：{s['accuracy']:.1%}（{s['observations']} 筆觀測）")
        add("")

    add("【行為輸出】每個身分各自的活動摘要（秒）與穩定度")
    for i, p in m["per_identity"].items():
        dist = "、".join(f"{k} {v:g}s" for k, v in p["behavior_seconds"].items()) or "（無）"
        add(f"  身分 {i}：{dist}")
        add(
            f"    最終輸出：{p['behavior_segments']} 段，其中短於 {SHORT_SEGMENT_SECONDS:g} 秒的 {p['short_behavior_segments']} 段，"
            f"每分鐘切換 {p['behavior_changes_per_minute']:.1f} 次"
        )
        add(f"    原始判斷（未平滑）：{p['raw_behavior_segments']} 段，其中短於 {SHORT_SEGMENT_SECONDS:g} 秒的 {p['raw_short_behavior_segments']} 段")
    add("　（行為統計不含 analyzing：那是追蹤剛出現、資料還不夠的暫時標籤，不是行為）")
    add("")

    if m.get("log"):
        add("【行為紀錄】實際寫進 txt / 之後會上傳的區段（依追蹤編號）")
        for pid, s in m["log"].items():
            dist = "、".join(f"{k} {v:g}s" for k, v in s["seconds"].items()) or "（無）"
            add(f"  Pet {pid}：{s['segments']} 段，其中短於 {SHORT_SEGMENT_SECONDS:g} 秒的 {s['short_segments']} 段；{dist}")
            for seg in s.get("timeline", []):
                add(f"      {clock(seg['start'])} - {clock(seg['end'])}  {seg['action']}")
        add("")

    add("【判讀】（門檻僅供參考）")
    for note in build_verdict(m):
        add(f"  {note}")

    return "\n".join(lines)


def git_revision():
    def run(*cmd):
        return subprocess.run(["git", *cmd], cwd=REPO_DIR, capture_output=True, text=True).stdout.strip()

    try:
        rev = run("rev-parse", "--short", "HEAD") or "unknown"
        dirty = run("status", "--porcelain", "--", "main_model.py")
        return f"{rev}{'（main_model.py 有未提交的變更）' if dirty else ''}"
    except OSError:
        return "unknown"


# ============================================================
# 主流程 / 自我測試
# ============================================================

def synth_single_gt(video_path):
    """單隻寵物的影片：整個畫面都算身分 A，只用來量行為穩定度與偵測 / 追蹤覆蓋率（身分指標沒有意義）。"""
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return {
        "mode": "single", "fps": fps, "frame_count": n, "panel_width": w + 1,
        "identities": {"A": {}},
        "segments": [{"start_frame": 0, "end_frame": n - 1, "left": "A", "right": "A"}],
    }


def apply_overrides(main_model, items):
    """--set NAME=VALUE：臨時覆寫 main_model 的模組常數（不改檔案），用來掃描參數。"""
    applied = {}
    for item in items or []:
        name, _, raw = item.partition("=")
        if not name or not raw:
            sys.exit(f"--set 格式應為 NAME=VALUE，收到：{item}")
        if not hasattr(main_model, name):
            sys.exit(f"main_model 沒有這個常數：{name}")
        value = ast.literal_eval(raw)
        setattr(main_model, name, value)
        applied[name] = value
    return applied


def evaluate(args):
    video_path = Path(args.video).resolve()
    if not video_path.exists():
        sys.exit(f"找不到影片：{video_path}")

    unlabeled = args.unlabeled
    gt = synth_single_gt(video_path) if (args.single or unlabeled) else load_gt(video_path, args.gt)
    main_model = load_main_model()
    overrides = apply_overrides(main_model, args.set)
    frames_instances = get_instances(main_model, video_path, args.refresh)

    tag = f"_{args.tag}" if args.tag else ""
    debug_path = EVAL_DIR / f"{video_path.stem}{tag}_eval.mp4" if args.debug_video else None

    t0 = time.time()
    if args.stride > 1 and not (args.single or unlabeled):
        sys.exit("--stride 只能搭配 --single 或 --unlabeled（有 ground truth 的拼接影片，幀編號對不上）")

    records, fps, log_segments, instances = run_tracker(
        main_model, video_path, frames_instances, gt, debug_path, args.max_frames, args.stride)
    print(f"追蹤 + 行為判斷完成，用時 {time.time() - t0:.0f} 秒")

    if unlabeled:
        metrics = compute_unlabeled_metrics(records, instances, fps, args.expected_count)
    else:
        metrics = compute_metrics(records, gt, fps)
        metrics["log"] = summarize_log(log_segments, fps)

    meta = {
        "video": video_path.name,
        "mode": gt.get("mode", "?"),
        "fps": fps,
        "git": git_revision(),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tracker_params": {k: getattr(main_model, k) for k in TRACKER_PARAMS if hasattr(main_model, k)},
        "overrides": overrides,
        "stride": args.stride,
    }

    report = format_unlabeled_report(meta, metrics) if unlabeled else format_report(meta, metrics)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    base = EVAL_DIR / f"{video_path.stem}{tag}_report"
    base.with_suffix(".txt").write_text(report, encoding="utf-8")
    base.with_suffix(".json").write_text(
        json.dumps({"meta": meta, "metrics": metrics}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print()
    print(report)
    print()
    print(f"報告：{base.with_suffix('.txt')}")
    if debug_path:
        label = "U<追蹤實例編號>，細白框是偵測器給的框" if (unlabeled or args.single) else "T<追蹤編號>=<真實身分>"
        print(f"除錯影片：{debug_path}（每個點標示 {label}）")


def _synthetic_records(n, segments, plan):
    """plan(frame) -> [(pid, x, action)]；det_true 與 obs 依 ground truth 自動標記。"""
    gt = {"identities": {"A": {}, "B": {}}, "panel_width": 100, "segments": segments}
    records = []
    for f in range(n):
        obs = [{"pid": pid, "true": true_identity(gt, f, x), "x": x, "y": 50, "action": action, "raw_action": action}
               for pid, x, action in plan(f)]
        records.append({"frame": f, "det_true": [o["true"] for o in obs], "obs": obs})
    return gt, records


def selftest():
    def close(a, b):
        return abs(a - b) < 1e-9

    one = [{"start_frame": 0, "end_frame": 99, "left": "A", "right": "B"}]
    two = [{"start_frame": 0, "end_frame": 49, "left": "A", "right": "B"},
           {"start_frame": 50, "end_frame": 99, "left": "B", "right": "A"}]

    # 1. 完美追蹤
    gt, rec = _synthetic_records(100, one, lambda f: [(1, 50, "sitting"), (2, 150, "walking")])
    m = compute_metrics(rec, gt, 30.0)
    assert close(m["idf1"], 1.0), m["idf1"]
    assert all(p["id_switches"] == 0 and len(p["track_ids_used"]) == 1 for p in m["per_identity"].values())
    assert not m["identity_changed_tracks"]
    assert m["per_identity"]["A"]["behavior_segments"] == 1

    # 2. 編號跟著「畫面位置」走（對調後兩隻動物換邊，但編號沒跟著動物走）
    gt, rec = _synthetic_records(100, two, lambda f: [(1, 50, "sitting"), (2, 150, "walking")])
    m = compute_metrics(rec, gt, 30.0)
    assert close(m["idf1"], 0.5), m["idf1"]
    assert {t["pid"] for t in m["identity_changed_tracks"]} == {1, 2}
    assert m["per_identity"]["A"]["id_switches"] == 1 and m["per_identity"]["B"]["id_switches"] == 1

    # 3. 編號「跟著動物」（對調後編號隨動物換邊）→ 應該是完美
    gt, rec = _synthetic_records(
        100, two, lambda f: [(1, 50 if f < 50 else 150, "sitting"), (2, 150 if f < 50 else 50, "walking")])
    m = compute_metrics(rec, gt, 30.0)
    assert close(m["idf1"], 1.0), m["idf1"]
    assert not m["identity_changed_tracks"]

    # 4. 破碎：A 在第 50 幀被換成新編號 3
    gt, rec = _synthetic_records(100, one, lambda f: [(1 if f < 50 else 3, 50, "sitting"), (2, 150, "walking")])
    m = compute_metrics(rec, gt, 30.0)
    assert close(m["idf1"], 0.75), m["idf1"]
    assert m["per_identity"]["A"]["track_ids_used"] == [1, 3] and m["per_identity"]["A"]["id_switches"] == 1
    assert m["per_identity"]["B"]["id_switches"] == 0

    # 5. 漏偵測：B 只有前 60 幀有追蹤 → 追蹤覆蓋率 60%，召回率下降
    gt, rec = _synthetic_records(100, one, lambda f: [(1, 50, "sitting")] + ([(2, 150, "walking")] if f < 60 else []))
    m = compute_metrics(rec, gt, 30.0)
    assert close(m["per_identity"]["B"]["tracked_frame_ratio"], 0.6)
    assert close(m["idr"], 160 / 200)

    # 6. 行為閃動：每 3 幀（0.1 秒）換一次標籤
    gt, rec = _synthetic_records(
        99, one, lambda f: [(1, 50, "standing" if (f // 3) % 2 == 0 else "walking"), (2, 150, "sitting")])
    m = compute_metrics(rec, gt, 30.0)
    assert m["per_identity"]["A"]["behavior_segments"] == 33
    assert m["per_identity"]["A"]["short_behavior_segments"] == 33

    # 7. 重複追蹤：同一隻被 2 個追蹤佔住，不應被誤算成 ID 切換
    gt, rec = _synthetic_records(100, one, lambda f: [(1, 50, "sitting"), (5, 55, "sitting"), (2, 150, "walking")])
    m = compute_metrics(rec, gt, 30.0)
    assert m["per_identity"]["A"]["id_switches"] == 0
    assert m["per_identity"]["A"]["track_duplicate_frames"] == 100

    # 8. 行為紀錄摘要：analyzing 不算行為；短區段要被數出來
    log = {1: [{"action": "analyzing", "start_frame": 0, "end_frame": 2},
               {"action": "sitting", "start_frame": 3, "end_frame": 62},
               {"action": "walking", "start_frame": 63, "end_frame": 72}]}
    s = summarize_log(log, 30.0)
    assert s[1]["segments"] == 2 and s[1]["short_segments"] == 1, s
    assert close(s[1]["seconds"]["sitting"], 2.0), s

    # 9. 無標註多隻：一條長壽、一條中途加入、一條 10 幀的碎片、一條從未確認且重用編號
    def inst(pid, born, confirmed_at, last_seen, seen):
        return {"pid": pid, "born": born, "confirmed_at": confirmed_at, "last_seen": last_seen, "seen_frames": seen}

    recs = []
    for f in range(100):
        obs = [{"pid": 1, "uid": 1, "action": "sitting"}]
        if f >= 20:
            obs.append({"pid": 2, "uid": 2, "action": "walking"})
        if 50 <= f < 60:
            obs.append({"pid": 3, "uid": 3, "action": "standing"})
        recs.append({"frame": f, "n_det": len(obs), "obs": obs})

    instances = {1: inst(1, 0, 4, 99, 96), 2: inst(2, 20, 24, 99, 76), 3: inst(3, 50, 54, 59, 10), 4: inst(3, 70, None, 72, 3)}
    m = compute_unlabeled_metrics(recs, instances, 30.0, expected_count=2)
    assert m["tracks_created"] == 4 and m["tracks_confirmed"] == 3 and m["tracks_never_confirmed"] == 1, m
    assert m["fragment_tracks"] == [3], m["fragment_tracks"]       # 只有 10 幀（0.2 秒）那條
    assert m["reused_pids"] == {"3": 2}, m["reused_pids"]
    assert m["extra_tracks"] == 1 and m["confirmed_max"] == 3, m
    assert close(m["frames_with_expected_count"], 0.7) and close(m["frames_over_count"], 0.1), m
    assert close(m["frames_under_count"], 0.2), m
    assert m["late_confirmed_tracks"] == 1, m  # 只有第 54 幀那條是在 1 秒（30 幀）之後才確認

    print("selftest：9 個情境全部通過")


def main():
    ap = argparse.ArgumentParser(description="追蹤 / 身分穩定度評估")
    ap.add_argument("video", nargs="?", help="要評估的影片（旁邊要有同名的 .gt.json）")
    ap.add_argument("--gt", default=None, help="ground truth 路徑，預設 <影片>.gt.json")
    ap.add_argument("--refresh", action="store_true", help="忽略快取，重跑 DeepLabCut")
    ap.add_argument("--debug-video", action="store_true", help="輸出標示追蹤編號與真實身分的除錯影片")
    ap.add_argument("--tag", default="", help="報告檔名後綴，例如 baseline / after-fix，方便前後比較")
    ap.add_argument("--max-frames", type=int, default=None, help="只評估前 N 幀（快速測試用）")
    ap.add_argument("--single", action="store_true", help="單隻寵物的影片（不需要 .gt.json）：只看偵測覆蓋與行為穩定度")
    ap.add_argument("--stride", type=int, default=1,
                    help="只處理每第 N 幀（模擬抽樣分析，例如 3 = 30 FPS 影片當 10 FPS 分析）；需搭配 --single / --unlabeled")
    ap.add_argument("--unlabeled", action="store_true",
                    help="多隻、沒有 ground truth 的真實影片：量偵測 / 追蹤數量、碎片、編號重複使用、行為閃動")
    ap.add_argument("--expected-count", type=int, default=None, help="--unlabeled 用：影片裡實際有幾隻（有的話能算多出幾條追蹤）")
    ap.add_argument("--set", action="append", metavar="NAME=VALUE", help="臨時覆寫 main_model 的常數，可重複，例如 --set MIN_ACTION_SECONDS=0.5")
    ap.add_argument("--selftest", action="store_true", help="用合成資料驗證指標計算本身，不需要 DeepLabCut")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if not args.video:
        ap.error("請指定影片，或使用 --selftest")

    evaluate(args)


if __name__ == "__main__":
    main()
