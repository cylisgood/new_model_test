"""
多寵物辨認（pet_identity.py）的單元測試：環境變數、種類排除、外觀比對與信心門檻、
取樣到判斷的流程、匯出器只上傳認得出的區段、大頭照只從 Supabase 網域抓。
不連網路、不需要攝影機或 GPU，也不載入 DeepLabCut / torch 權重（分類器用假的）。

用法（在專案根目錄）：
  venv\\Scripts\\python.exe evaluation\\test_pet_identity.py
"""

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
import behavior_export as be  # noqa: E402
import pet_identity as pi  # noqa: E402

DOG_A = "11111111-1111-1111-1111-111111111111"
DOG_B = "22222222-2222-2222-2222-222222222222"
CAT_C = "33333333-3333-3333-3333-333333333333"


def quiet(_message):
    pass


def vec(*values):
    return pi._unit(np.array(values, dtype=np.float32))


def hist_of(seed):
    rng = np.random.default_rng(seed)
    return rng.random((16, 16)).astype(np.float32)


def sample(species="dog", conf=0.95, embed=None, hist=None, frame=0):
    return {"frame": frame, "species": species, "species_conf": conf, "embed": embed, "hist": hist}


def profile(pet_id, species, embed=None, hist=None, name=None):
    return pi.Profile(pet_id, name or pet_id[:2], species, embed, hist)


# ---- 環境變數 ----

def test_identify_mode():
    assert pi.read_identify_mode({}) is False
    assert pi.read_identify_mode({"PET_IDENTIFY": "off"}) is False
    assert pi.read_identify_mode({"PET_IDENTIFY": "ON"}) is True
    try:
        pi.read_identify_mode({"PET_IDENTIFY": "yes please"})
        raise AssertionError("打錯字應該報錯")
    except ValueError:
        pass


def test_pet_ids_and_thresholds():
    assert pi.read_pet_ids({}) is None
    assert pi.read_pet_ids({"PET_IDS": f"{DOG_A.upper()}, {CAT_C}"}) == [DOG_A, CAT_C]
    try:
        pi.read_pet_ids({"PET_IDS": "not-a-uuid"})
        raise AssertionError("不是 uuid 應該報錯")
    except ValueError:
        pass

    assert pi.read_thresholds({}) == (pi.DEFAULT_MIN_SCORE, pi.DEFAULT_MIN_MARGIN, pi.DEFAULT_SAMPLE_EVERY)
    assert pi.read_thresholds({"PET_ID_MIN_SCORE": "0.7", "PET_ID_MIN_MARGIN": "0.2", "PET_ID_SAMPLE_EVERY": "2"}) == (0.7, 0.2, 2)
    try:
        pi.read_thresholds({"PET_ID_SAMPLE_EVERY": "0"})
        raise AssertionError("取樣間隔不能小於 1")
    except ValueError:
        pass


# ---- 判斷 ----

def test_species_excludes_other_species():
    e = vec(1, 0, 0)
    d = pi.decide([sample("cat", embed=e)] * 5, [profile(DOG_A, "dog", e), profile(DOG_B, "dog", vec(0, 1, 0))])
    assert d.pet_id is None and d.reason == "no_species_match", d


def test_single_candidate_of_species_is_accepted():
    # 一狗一貓：看到狗就是那隻狗，不需要外觀
    d = pi.decide([sample("dog")] * 4, [profile(DOG_A, "dog"), profile(CAT_C, "cat")])
    assert d.pet_id == DOG_A and d.reason == "ok", d
    d = pi.decide([sample("cat")] * 4, [profile(DOG_A, "dog"), profile(CAT_C, "cat")])
    assert d.pet_id == CAT_C, d


def test_two_dogs_pick_closer_appearance():
    a, b = vec(1, 0.1, 0), vec(0, 0.1, 1)
    profiles = [profile(DOG_A, "dog", a, hist_of(1)), profile(DOG_B, "dog", b, hist_of(2))]
    near_a = [sample("dog", embed=vec(1, 0.15, 0.05), hist=hist_of(1))] * 6
    near_b = [sample("dog", embed=vec(0.05, 0.1, 1), hist=hist_of(2))] * 6

    assert pi.decide(near_a, profiles).pet_id == DOG_A
    assert pi.decide(near_b, profiles).pet_id == DOG_B


def test_two_lookalike_dogs_are_not_guessed():
    same = vec(1, 1, 0)
    profiles = [profile(DOG_A, "dog", same, hist_of(1)), profile(DOG_B, "dog", same, hist_of(1))]
    d = pi.decide([sample("dog", embed=same, hist=hist_of(1))] * 6, profiles)
    assert d.pet_id is None and d.reason == "ambiguous", d


def test_low_score_is_rejected():
    profiles = [profile(DOG_A, "dog", vec(1, 0, 0), hist_of(1)), profile(DOG_B, "dog", vec(0, 1, 0), hist_of(2))]
    far = [sample("dog", embed=vec(0, 0, 1), hist=hist_of(3))] * 6
    d = pi.decide(far, profiles, min_score=0.95)
    assert d.pet_id is None and d.reason == "low_score", d


def test_missing_reference_with_several_candidates_is_rejected():
    profiles = [profile(DOG_A, "dog", vec(1, 0, 0), hist_of(1)), profile(DOG_B, "dog")]   # B 沒大頭照
    d = pi.decide([sample("dog", embed=vec(1, 0, 0), hist=hist_of(1))] * 6, profiles)
    assert d.pet_id is None and d.reason == "missing_reference", d


def test_not_enough_or_unclear_samples():
    profiles = [profile(DOG_A, "dog"), profile(CAT_C, "cat")]
    assert pi.decide([sample("dog")] * (pi.MIN_SAMPLES - 1), profiles).reason == "too_few_samples"
    assert pi.decide([sample(None, 0.0)] * 5, profiles).reason == "species_unknown"
    # 一半投狗、一半投貓：種類不確定，不猜
    assert pi.decide([sample("dog")] * 3 + [sample("cat")] * 3, profiles).reason == "species_ambiguous"
    assert pi.decide([sample("dog")] * 5, []).reason == "no_profiles"


def test_species_vote_is_confidence_weighted():
    species, share, _, _ = pi.aggregate_samples([sample("dog", 0.9)] * 3 + [sample("cat", 0.2)] * 3)
    assert species == "dog" and share > 0.8


def test_profile_without_species_needs_score():
    e = vec(1, 0, 0)
    d = pi.decide([sample("dog", embed=e, hist=hist_of(1))] * 5, [profile(DOG_A, None, e, hist_of(1))])
    assert d.pet_id == DOG_A, d
    d = pi.decide([sample("dog", embed=vec(0, 1, 0), hist=hist_of(9))] * 5, [profile(DOG_A, None, e, hist_of(1))], min_score=0.95)
    assert d.pet_id is None and d.reason == "low_score", d


# ---- 辨認器（取樣 → 判斷）----

class FakeEncoder:
    """第一個像素通道的平均值決定種類：偏亮 = 狗、偏暗 = 貓。"""

    def encode_many(self, crops):
        out = []
        for crop in crops:
            species = "dog" if crop.mean() > 100 else "cat"
            out.append((species, 0.9, vec(1, float(crop.mean()) / 255.0, 0)))
        return out


def frame_with(value):
    return np.full((120, 160, 3), value, dtype=np.uint8)


def test_identifier_samples_and_resolves_per_segment():
    identifier = pi.PetIdentifier([profile(DOG_A, "dog"), profile(CAT_C, "cat")], FakeEncoder(), sample_every=2)
    bbox = (10, 10, 100, 100)

    assert identifier.will_sample(0) and not identifier.will_sample(1)

    for f in range(0, 20):
        identifier.observe(7, f, frame_with(200), bbox)       # 追蹤 7 在 0~19 幀是狗
    for f in range(20, 40):
        identifier.observe(7, f, frame_with(30), bbox)        # 同一個追蹤編號後來變成貓（編號互換）

    assert identifier.resolve(7, {"start_frame": 0, "end_frame": 19}).pet_id == DOG_A
    assert identifier.resolve(7, {"start_frame": 20, "end_frame": 39}).pet_id == CAT_C
    # 用過的取樣會被丟掉，不會一直累積
    assert identifier.resolve(7, {"start_frame": 0, "end_frame": 39}).reason == "too_few_samples"
    assert identifier.resolve(99, {"start_frame": 0, "end_frame": 5}).reason == "too_few_samples"


def test_observe_ignores_tiny_or_missing_bbox():
    identifier = pi.PetIdentifier([profile(DOG_A, "dog")], FakeEncoder(), sample_every=1)
    identifier.observe(1, 0, frame_with(200), None)
    identifier.observe(1, 1, frame_with(200), (5, 5, 10, 10))
    assert identifier.resolve(1, {"start_frame": 0, "end_frame": 5}).reason == "too_few_samples"


# ---- 匯出器 ----

def fixed_clock():
    return datetime(2026, 11, 4, 12, 0, 0, tzinfo=timezone.utc)


class FakeUploader:
    def __init__(self):
        self.batches = []

    def submit(self, rows):
        self.batches.append(list(rows))

    def close(self, timeout=10.0):
        pass


def make_resolver(mapping):
    def resolver(track_id, segment):
        pet = mapping.get(track_id)
        return pi.Decision(pet, "ok" if pet else "ambiguous", {}, 5)

    return resolver


def segments():
    return {
        1: [{"action": "walking", "start_frame": 0, "end_frame": 60}],
        2: [{"action": "lying", "start_frame": 0, "end_frame": 60}],
    }


def test_exporter_writes_resolved_pet_ids_and_keeps_unidentified_locally():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.jsonl"
        exporter = be.DryRunExporter(path, clock=fixed_clock, resolver=make_resolver({1: DOG_A}))
        assert exporter.export_new(segments(), fps=30) == 2

        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        by_track = {r["_track_id"]: r for r in rows}
        assert by_track[1]["pet_id"] == DOG_A and by_track[1]["_identity"]["reason"] == "ok"
        assert by_track[2]["pet_id"] is None and by_track[2]["_identity"]["reason"] == "ambiguous"
        assert exporter.unidentified == 1


def test_supabase_exporter_uploads_only_identified_rows():
    with tempfile.TemporaryDirectory() as tmp:
        uploader = FakeUploader()
        exporter = be.SupabaseExporter(
            Path(tmp) / "out.jsonl", uploader, None, clock=fixed_clock, resolver=make_resolver({1: DOG_A, 2: CAT_C})
        )
        exporter.export_new(segments(), fps=30)
        assert len(uploader.batches) == 1
        assert {r["pet_id"] for r in uploader.batches[0]} == {DOG_A, CAT_C}

        uploader2 = FakeUploader()
        exporter2 = be.SupabaseExporter(Path(tmp) / "out2.jsonl", uploader2, None, clock=fixed_clock, resolver=make_resolver({}))
        exporter2.export_new(segments(), fps=30)
        assert uploader2.batches == [], "全部認不出時不能上傳任何東西"


def test_exporter_without_resolver_behaves_as_before():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.jsonl"
        exporter = be.DryRunExporter(path, DOG_A, clock=fixed_clock)
        exporter.export_new(segments(), fps=30)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert {r["pet_id"] for r in rows} == {DOG_A}
        assert all("_identity" not in r for r in rows)


# ---- 大頭照 ----

def test_avatar_only_from_supabase_host():
    for bad in ("http://proj.supabase.co/a.png", "https://evil.example.com/a.png", "file:///etc/passwd", "", None):
        try:
            pi._fetch_avatar(bad, "proj.supabase.co", opener=lambda *a, **k: None)
            raise AssertionError(f"不該抓：{bad!r}")
        except ValueError:
            pass


def test_build_profiles_handles_missing_avatar_and_skips_unusable():
    import cv2

    ok, png = cv2.imencode(".png", frame_with(200))
    assert ok

    def fetch(url, host):
        if "broken" in url:
            raise ValueError("壞掉")
        return frame_with(200)

    pets = [
        {"id": DOG_A, "name": "掰掰", "species": "dog", "avatar_url": "https://h/ok.png"},
        {"id": DOG_B, "name": "壞圖", "species": "dog", "avatar_url": "https://h/broken.png"},
        {"id": CAT_C, "name": "無資料", "species": None, "avatar_url": None},
    ]
    messages = []
    profiles = pi.build_profiles(pets, FakeEncoder(), "h", fetch=fetch, log=messages.append)

    assert [p.pet_id for p in profiles] == [DOG_A, DOG_B]
    assert profiles[0].embed is not None and profiles[1].embed is None
    assert any("讀不到" in m for m in messages) and any("略過" in m for m in messages)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"通過：{t.__name__}")
    print(f"\n{len(tests)} 個測試全部通過")
