"""
降低分析取樣（ANALYSIS_FPS）相關的單元測試：依時間挑畫面（FrameSampler）、
追蹤確認與消失保留改成以秒計（track_min_hits / max_missing_frames），以及高幀率時維持原本的行為。
不連網路、不需要攝影機或 GPU；匯入 main_model 會載入 DeepLabCut，需要比較久。

用法（在專案根目錄）：
  venv\\Scripts\\python.exe evaluation\\test_analysis_sampling.py
"""

import os
import sys
from pathlib import Path

os.environ["SKIP_ENV_FILE"] = "1"   # 不讀 .env，測試不需要任何帳密

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
import main_model as mm  # noqa: E402


def simulate(source_fps, seconds, interval, jitter=0.0):
    """模擬來源以 source_fps 送畫面，回傳被取走的時間點。"""
    now = [0.0]
    sampler = mm.FrameSampler(interval, clock=lambda: now[0])
    taken = []
    n = int(source_fps * seconds)
    for i in range(n):
        now[0] = i / source_fps + (jitter if i % 2 else 0.0)
        if sampler.take():
            taken.append(now[0])
    return taken


def test_sampler_takes_every_frame_when_disabled():
    assert len(simulate(20, 10, 0)) == 200


def test_sampler_rate_matches_target():
    for source in (20, 25, 30, 50):
        taken = simulate(source, 30, 1 / 3)
        rate = len(taken) / 30
        assert abs(rate - 3.0) < 0.1, (source, rate)


def test_sampler_rate_survives_arrival_jitter():
    taken = simulate(20, 30, 1 / 3, jitter=0.02)
    assert abs(len(taken) / 30 - 3.0) < 0.15, len(taken) / 30


def test_sampler_spacing_is_even():
    taken = simulate(20, 20, 1 / 3)
    gaps = [b - a for a, b in zip(taken, taken[1:])]
    assert min(gaps) >= 0.25 and max(gaps) <= 0.40, (min(gaps), max(gaps))


def test_sampler_recovers_after_a_gap():
    now = [0.0]
    sampler = mm.FrameSampler(0.5, clock=lambda: now[0])
    assert sampler.take()
    now[0] = 100.0          # 斷線很久之後
    assert sampler.take()   # 立刻取一張
    now[0] = 100.1
    assert not sampler.take()   # 之後不會連續補取一大堆
    now[0] = 100.6
    assert sampler.take()


def test_sampler_reset():
    now = [0.0]
    sampler = mm.FrameSampler(1.0, clock=lambda: now[0])
    assert sampler.take()
    now[0] = 0.1
    assert not sampler.take()
    sampler.reset()
    assert sampler.take()


def test_track_min_hits_keeps_old_value_at_high_fps():
    for fps in (30, 25, 20, 15, 10):
        assert mm.track_min_hits(fps) == mm.TRACK_MIN_HITS, fps


def test_track_min_hits_shrinks_at_low_fps():
    assert mm.track_min_hits(3) == 2
    assert mm.track_min_hits(2) == 2     # 至少 2 幀，不會一幀就確認
    assert mm.track_min_hits(5) == 4
    # 3 張／秒時確認時間不到 1 秒，不是原本逐幀值換算出的 1.7 秒
    assert mm.track_min_hits(3) / 3 < 1.0


def test_max_missing_is_about_three_seconds():
    assert mm.max_missing_frames(30) == mm.MAX_MISSING_FRAMES
    assert abs(mm.max_missing_frames(3) / 3 - mm.MAX_MISSING_SECONDS) < 0.4
    assert mm.max_missing_frames(1) >= mm.TENTATIVE_MAX_MISSING
    assert mm.max_missing_frames(120) <= mm.MAX_MISSING_FRAMES


def det(center=(100, 100)):
    return {"center": center, "neck": (90, 100), "tail": (110, 100), "body_len": 20.0, "body_size": 30.0,
            "color_hist": None, "features": {}, "front_paw": None, "back_paw": None,
            "hip_to_back_paw": None, "paw_drop": None, "bbox_aspect": None}


def test_new_track_confirms_after_two_hits_at_3fps_and_five_at_30fps():
    low = mm.create_new_track(det(), 0, fps=3.0)
    assert not low["confirmed"] and low["min_hits"] == 2
    mm.update_track(low, det(), 1)
    assert low["confirmed"]

    high = mm.create_new_track(det(), 0, fps=30.0)
    for i in range(1, 4):
        mm.update_track(high, det(), i)
    assert not high["confirmed"]
    mm.update_track(high, det(), 4)
    assert high["confirmed"]


def test_history_frames_keeps_old_value_at_high_fps_and_shrinks_at_low_fps():
    assert mm.history_frames(30) == 30     # 約 1 秒，跟以前一樣
    assert mm.history_frames(10) == 10
    assert mm.history_frames(20) == 20
    # 低幀率不再被撐成 2.7 秒：3 張／秒只留 4 幀（約 1.3 秒）
    assert mm.history_frames(3) == 4
    assert mm.history_frames(3) / 3 < 1.5
    assert mm.history_frames(1) == 4


def test_ema_alpha_keeps_old_value_at_high_fps_and_follows_more_closely_at_low_fps():
    assert mm.ema_alpha(30) == 0.4
    assert mm.ema_alpha(10) == 0.4
    assert mm.ema_alpha(5) > 0.4
    assert 0.8 < mm.ema_alpha(3) <= 1.0
    assert mm.ema_alpha(1) <= 1.0


def test_new_track_uses_low_fps_history_and_smoothing():
    low = mm.create_new_track(det(), 0, fps=3.0)
    assert low["history"].maxlen == 4 and low["ema_alpha"] == mm.ema_alpha(3.0)
    high = mm.create_new_track(det(), 0, fps=30.0)
    assert high["history"].maxlen == 30 and high["ema_alpha"] == 0.4


def test_low_fps_track_follows_a_jump_instead_of_lagging():
    # 低取樣率下一幀移動很遠，平滑後的位置不該只走一小部分
    low = mm.create_new_track(det((100, 100)), 0, fps=3.0)
    mm.update_track(low, det((400, 100)), 1)
    assert low["ema_center"][0] > 300     # 跟上了大部分的跳動（舊的 0.4 只會走到約 220）
    high = mm.create_new_track(det((100, 100)), 0, fps=30.0)
    mm.update_track(high, det((400, 100)), 1)
    assert high["ema_center"][0] < 250     # 高幀率維持原本的平滑


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"通過：{t.__name__}")
    print(f"\n{len(tests)} 個測試全部通過")
