"""
行為標籤去抖動（stabilize_action + update_behavior_segments）的單元測試。
用合成的逐幀「原始判斷」序列驅動，不需要 DeepLabCut 推論，也不需要任何影片。

用法（在專案根目錄）：
  venv\\Scripts\\python.exe evaluation\\test_behavior_stabilization.py
"""

import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main_model as mm  # noqa: E402  (會載入 deeplabcut，要幾秒)


def run(raw_labels, fps=30.0):
    """把一串逐幀原始判斷餵進去，回傳 (最終每幀標籤, 行為區段列表)。"""
    track = {"action_history": deque(maxlen=mm.seconds_to_frames(mm.ACTION_VOTE_SECONDS, fps, 3))}
    behavior_segments, active_segments = {}, {}
    per_frame = []

    for frame_id, raw in enumerate(raw_labels):
        track["action_history"].append(raw)
        action, switch_frame = mm.stabilize_action(track, frame_id, fps)
        mm.update_behavior_segments(behavior_segments, active_segments, 1, action, frame_id, switch_frame)
        per_frame.append(action)

    segments = behavior_segments.get(1, []) + [dict(active_segments[1])]
    return per_frame, segments


def check_partition(segments, n_frames):
    """區段必須完整、不重疊、不留空隙地涵蓋所有幀（回溯切換時最容易出錯的地方）。"""
    assert segments[0]["start_frame"] == 0, segments
    assert segments[-1]["end_frame"] == n_frames - 1, segments
    for a, b in zip(segments, segments[1:]):
        assert a["end_frame"] + 1 == b["start_frame"], f"區段不連續：{a} -> {b}"
    for s in segments:
        assert s["start_frame"] <= s["end_frame"], s


def test_brief_burst_is_absorbed():
    # 站 40 幀 → 走 20 幀（0.67 秒，不到 1 秒）→ 站 40 幀：短暫的走路不該進紀錄
    raw = ["standing"] * 40 + ["walking"] * 20 + ["standing"] * 40
    per_frame, segs = run(raw)
    check_partition(segs, len(raw))
    assert [s["action"] for s in segs] == ["standing"], segs
    assert set(per_frame) == {"standing"}


def test_sustained_change_is_kept_and_backdated():
    # 站 40 → 走 60（2 秒）→ 站 40：走路要被承認，而且起點要回溯到「真正開始」而不是延遲 1 秒
    raw = ["standing"] * 40 + ["walking"] * 60 + ["standing"] * 40
    per_frame, segs = run(raw)
    check_partition(segs, len(raw))
    assert [s["action"] for s in segs] == ["standing", "walking", "standing"], segs

    walk = segs[1]
    # 原始判斷在第 40 幀開始走；多數決視窗約晚 5~6 幀轉向，回溯後應該落在 40~50 之間，
    # 若沒有回溯，起點會是「確認當下」≈ 第 75 幀以後
    assert 40 <= walk["start_frame"] <= 50, walk
    assert 100 <= walk["end_frame"] <= 110, walk


def test_tie_does_not_flip():
    # 兩種標籤逐幀交替（視窗內永遠平手）：應該維持原本的行為，不跟著 Counter 順序亂跳
    raw = ["sitting"] * 30 + ["sitting", "standing"] * 30
    per_frame, segs = run(raw)
    check_partition(segs, len(raw))
    assert [s["action"] for s in segs] == ["sitting"], segs


def test_analyzing_commits_immediately():
    # analyzing 是「資料不足」的暫時標籤，不該被去抖動拖住 1 秒
    raw = ["analyzing"] * 3 + ["sitting"] * 60
    per_frame, segs = run(raw)
    check_partition(segs, len(raw))
    assert [s["action"] for s in segs] == ["analyzing", "sitting"], segs
    assert per_frame[12] == "sitting", per_frame[:15]  # 約第 6 個 sitting 之後多數就轉過去，不是等 30 幀


def test_fps_invariance():
    fps = 60.0
    assert mm.seconds_to_frames(mm.MIN_ACTION_SECONDS, fps) == 60

    # 60 FPS：走 50 幀 = 0.83 秒 → 被吸收；走 90 幀 = 1.5 秒 → 被承認
    short = ["standing"] * 80 + ["walking"] * 50 + ["standing"] * 80
    _, segs = run(short, fps)
    assert [s["action"] for s in segs] == ["standing"], segs

    long = ["standing"] * 80 + ["walking"] * 90 + ["standing"] * 120
    _, segs = run(long, fps)
    check_partition(segs, len(long))
    assert [s["action"] for s in segs] == ["standing", "walking", "standing"], segs


def test_update_behavior_segments_backdating():
    behavior_segments, active_segments = {}, {}
    for f in range(50):
        mm.update_behavior_segments(behavior_segments, active_segments, 1, "a", f)

    mm.update_behavior_segments(behavior_segments, active_segments, 1, "b", 50, switch_frame=30)

    assert behavior_segments[1] == [{"action": "a", "start_frame": 0, "end_frame": 29}], behavior_segments
    assert active_segments[1] == {"action": "b", "start_frame": 30, "end_frame": 50}, active_segments


def test_classify_action_fps_scaling():
    # 同樣的真實移動速度，在 30 FPS 與 60 FPS 下應該判斷成一樣的行為
    def history_for(fps, body_len=100.0, speed_bodylens_per_sec=2.0, seconds=1.0):
        step = speed_bodylens_per_sec * body_len / fps      # 每幀位移（像素）
        n = mm.seconds_to_frames(seconds, fps)
        return [
            {"center": (200 + step * i, 300), "neck": (200 + step * i, 250), "tail": (200 + step * i, 350),
             "body_len": body_len, "front_paw": (200 + step * i, 340), "back_paw": (200 + step * i, 345),
             "hip_to_back_paw": 0.6, "frame_id": i}
            for i in range(n)
        ]

    kpts = [[0, 0]] * len(mm.KEYPOINTS)
    scores = [0.0] * len(mm.KEYPOINTS)

    results = {}
    for fps in (15.0, 30.0, 60.0):
        history = history_for(fps)
        results[fps] = mm.classify_action(history, 1920, 1080, kpts, scores, camera_motion_cache=None, fps=fps)

    # 2 體長/秒 ≈ 每幀 0.067（30 FPS），高於 WALK_THRES=0.05 → 跑
    assert len(set(results.values())) == 1, results


def test_update_latest_result_uses_committed_action():
    # /current、/ws 讀的結果要跟行為紀錄一致：用去抖動後的行為，信心 = 最近逐幀判斷裡有幾成同意它
    tracks = {1: {"action_history": deque(["walking"] * 6 + ["sitting"] * 4), "current_action": "sitting"}}
    mm.update_latest_result(tracks)
    result = mm.latest_result
    assert result["behavior"] == "sitting" and result["confidence"] == 0.4, result
    assert result["summary"]["sitting"] == 1 and result["summary"]["walking"] == 0, result

    # 還沒有 current_action（舊資料）時，退回原本的多數決
    tracks = {1: {"action_history": deque(["walking"] * 6 + ["sitting"] * 4)}}
    mm.update_latest_result(tracks)
    assert mm.latest_result["behavior"] == "walking", mm.latest_result

    mm.mark_offline()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"通過：{t.__name__}")
    print(f"\n{len(tests)} 個測試全部通過")
