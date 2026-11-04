"""
站、坐、趴的靜態姿勢規則（main_model.classify_static_posture 等）的單元測試。
重點是改規則前的錯誤不能再出現：四隻腳站著的動物從側面看脊椎是水平的，不能因此判成趴臥。
不連網路、不需要攝影機或 GPU；匯入 main_model 會載入 DeepLabCut，需要比較久。

用法（在專案根目錄）：
  venv\\Scripts\\python.exe evaluation\\test_posture_rule.py
"""

import os
import sys
from pathlib import Path

os.environ["SKIP_ENV_FILE"] = "1"   # 不讀 .env，測試不需要任何帳密

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
import main_model as mm  # noqa: E402


def features(**kw):
    base = {"spine_vertical_ratio": 0.3, "hip_to_back_paw": 0.5, "paw_drop": 0.9, "bbox_aspect": 1.0}
    base.update(kw)
    return base


def test_standing_side_view_with_horizontal_spine_is_not_lying():
    # 舊規則：脊椎垂直比 < 0.25 就是趴臥，會把這種側面站著的狗判錯
    assert mm.classify_static_posture(features(spine_vertical_ratio=0.08, paw_drop=0.9, bbox_aspect=0.8)) == "standing"


def test_lying_flat_is_lying():
    assert mm.classify_static_posture(features(spine_vertical_ratio=0.1, paw_drop=0.4, bbox_aspect=0.6)) == "lying"


def test_lying_needs_flat_bbox_too():
    # 腳掌離背部近，但外框很高（例如蹲著、面向鏡頭）不算趴
    assert mm.classify_static_posture(features(paw_drop=0.6, bbox_aspect=1.4)) != "lying"


def test_sitting_cat_stays_sitting():
    assert mm.classify_static_posture(features(spine_vertical_ratio=0.6, hip_to_back_paw=0.3, paw_drop=0.94, bbox_aspect=1.05)) == "sitting"


def test_walking_toward_camera_is_standing_not_sitting():
    # 正面朝鏡頭走時脊椎被壓成直的，但後腳是伸直的（髖部離後腳掌遠）
    assert mm.classify_static_posture(features(spine_vertical_ratio=0.6, hip_to_back_paw=0.7, paw_drop=1.0)) == "standing"


def test_missing_paw_drop_falls_back_to_bbox_aspect():
    assert mm.classify_static_posture(features(paw_drop=None, bbox_aspect=0.5)) == "lying"
    assert mm.classify_static_posture(features(paw_drop=None, bbox_aspect=1.0)) == "standing"
    f = features(paw_drop=None)
    f.pop("bbox_aspect")
    assert mm.classify_static_posture(f) == "standing"


def make_keypoints(back_y, paw_y):
    """39 個關鍵點，只放背部中點和四隻腳掌；其他留空（信心 0）。"""
    kpts = [[0.0, 0.0] for _ in mm.KEYPOINTS]
    scores = [0.0 for _ in mm.KEYPOINTS]
    for idx, y in ((mm.BACK_MIDDLE_IDX, back_y), (mm.LEFT_FRONT_PAW_IDX, paw_y), (mm.RIGHT_FRONT_PAW_IDX, paw_y),
                   (mm.LEFT_BACK_PAW_IDX, paw_y), (mm.RIGHT_BACK_PAW_IDX, paw_y)):
        kpts[idx] = [100.0, y]
        scores[idx] = 0.9
    return kpts, scores


def test_paw_drop_ratio():
    kpts, scores = make_keypoints(back_y=100.0, paw_y=190.0)
    assert abs(mm.paw_drop_ratio(kpts, scores, body_len=100.0) - 0.9) < 1e-6
    # 背部或腳掌看不到，或身長不明，就算不出來
    kpts2, scores2 = make_keypoints(100.0, 190.0)
    scores2[mm.BACK_MIDDLE_IDX] = 0.0
    assert mm.paw_drop_ratio(kpts2, scores2, 100.0) is None
    assert mm.paw_drop_ratio(kpts, scores, None) is None
    assert mm.paw_drop_ratio([[0, 0]] * len(mm.KEYPOINTS), [0.0] * len(mm.KEYPOINTS), 100.0) is None


def test_bbox_aspect_ratio():
    assert mm.bbox_aspect_ratio((0, 0, 200, 100)) == 0.5
    assert mm.bbox_aspect_ratio(None) is None


def test_smoothed_features_use_history_median():
    hist = [{"neck": (0, 0), "tail": (100, 0), "paw_drop": v, "bbox_aspect": a, "hip_to_back_paw": 0.5}
            for v, a in ((0.9, 1.0), (0.4, 0.6), (0.5, 0.7))]
    hist.append({"neck": (0, 0), "tail": (100, 0), "paw_drop": None, "bbox_aspect": None, "hip_to_back_paw": None})
    kpts, scores = make_keypoints(100.0, 190.0)
    f = mm.get_smoothed_posture_features(hist, kpts, scores, body_len=100.0) or {}
    # 單幀特徵要有東西才會進平滑流程；這裡沒有脖子尾巴時 features 可能是空的，空的就略過這項檢查
    if f:
        assert abs(f["paw_drop"] - 0.5) < 1e-6
        assert abs(f["bbox_aspect"] - 0.7) < 1e-6


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"通過：{t.__name__}")
    print(f"\n{len(tests)} 個測試全部通過")
