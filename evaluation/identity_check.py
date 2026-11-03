"""
寵物辨認驗證工具：拿一段「多隻寵物同時入鏡」的影片，跑完整的 DeepLabCut → 追蹤 → 辨認流程，
印出每段行為被判成哪一隻、分數多少、認不出的原因，用來檢查認錯比例並校正門檻。
不連 Supabase、不上傳任何東西：參考照片從本機檔案讀。

用法（在專案根目錄）：
  venv\\Scripts\\python.exe evaluation\\identity_check.py 影片.mp4 ^
      --profile 小橘:cat:參考照1.jpg --profile 小白:cat:參考照2.jpg ^
      --left 小白 --right 小橘

  --profile 名字:種類:照片   重複給幾隻就比對幾隻；種類是 dog 或 cat，也可以寫 none
  --left / --right          選填。畫面左半邊、右半邊的動物是誰；有給就會對照「判斷結果」和「實際位置」，
                            算出認對、認錯、認不出的段數（適合 make_composite.py 拼出來的測試影片）
  --min-score / --min-margin / --sample-every   覆蓋門檻

限制：左右位置只能當作答案的前提是兩隻不會交換位置；真實的交錯、遮擋要用實拍影片人工檢查。
"""

import argparse
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

os.environ["SKIP_ENV_FILE"] = "1"   # 不讀 .env，這個工具不需要也不該碰任何帳密

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

import cv2  # noqa: E402

import main_model as mm  # noqa: E402
import pet_identity as pi  # noqa: E402


def parse_profile(text):
    parts = text.split(":", 2)

    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--profile 格式是 名字:種類:照片，例如 小橘:cat:orange.jpg")

    name, species, image_path = parts
    species = species.lower()

    if species not in ("dog", "cat", "none"):
        raise argparse.ArgumentTypeError("種類只能是 dog、cat 或 none")

    return name, (None if species == "none" else species), image_path


def build_local_profiles(specs, encoder):
    profiles = []

    for index, (name, species, image_path) in enumerate(specs):
        image = cv2.imread(image_path)

        if image is None:
            raise SystemExit(f"讀不到參考照片：{image_path}")

        h, w = image.shape[:2]
        dh, dw = int(h * (1 - pi.AVATAR_CENTER_RATIO) / 2), int(w * (1 - pi.AVATAR_CENTER_RATIO) / 2)
        center = image[dh:h - dh, dw:w - dw]
        (found, conf, embed), = encoder.encode_many([center])

        if species and found and found != species:
            print(f"⚠️ {name} 的照片看起來是 {found}（{conf:.2f}），但你指定的是 {species}")

        profiles.append(pi.Profile(f"pet-{index}", name, species, embed, pi.hsv_hist(center)))

    return profiles


def main():
    parser = argparse.ArgumentParser(description="寵物辨認驗證工具")
    parser.add_argument("video")
    parser.add_argument("--profile", action="append", type=parse_profile, required=True)
    parser.add_argument("--left")
    parser.add_argument("--right")
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--min-margin", type=float)
    parser.add_argument("--sample-every", type=int)
    args = parser.parse_args()

    default_score, default_margin, default_every = pi.read_thresholds()
    min_score = default_score if args.min_score is None else args.min_score
    min_margin = default_margin if args.min_margin is None else args.min_margin
    every = default_every if args.sample_every is None else args.sample_every

    work_dir = Path(tempfile.mkdtemp(prefix="identity_check_"))
    video = work_dir / "clip.mp4"
    shutil.copy(args.video, video)   # DeepLabCut 會在影片旁邊寫中間檔，不要弄髒原本的資料夾

    try:
        encoder = pi.CropEncoder()
        profiles = build_local_profiles(args.profile, encoder)
        identifier = pi.PetIdentifier(profiles, encoder, min_score, min_margin, every)

        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"影片 {width}x{height}，{fps:.1f} fps；門檻 min_score={min_score} min_margin={min_margin} 每 {every} 幀取樣")
        print("DeepLabCut 分析中（較慢，請等一下）...")
        frames_instances = mm.run_dlc_clip_inference(video)

        tracks, segments, active, cam = {}, {}, {}, {}
        centers = {}   # 追蹤編號 -> 每幀的 x 座標，用來對照左右位置
        prev_gray = None

        for frame_id, instances in enumerate(frames_instances):
            ok, frame = cap.read()

            if not ok:
                break

            _, dets, prev_gray = mm.process_behavior_frame(
                frame, frame_id, instances, tracks, segments, active, width, height, cam, prev_gray,
                fps=fps, identifier=identifier,
            )

            for track_id, track in tracks.items():
                if track["confirmed"] and track["missing_frames"] == 0 and track["last_center"] is not None:
                    centers.setdefault(track_id, []).append((frame_id, track["last_center"][0]))

        for track_id, segment in active.items():
            segments.setdefault(track_id, []).append(dict(segment))

        print(f"\n追蹤到 {len(segments)} 隻（追蹤編號，不是寵物身分）；每段行為的辨認結果：\n")

        outcome = Counter()
        names = {p.pet_id: p.name for p in profiles}

        for track_id, track_segments in sorted(segments.items()):
            for seg in track_segments:
                if seg["action"] not in ("lying", "walking", "running", "standing", "sitting"):
                    continue

                seconds = (seg["end_frame"] - seg["start_frame"]) / fps

                if seconds < 1.0:
                    continue

                decision = identifier.resolve(track_id, seg)
                got = names.get(decision.pet_id) if decision.pet_id else None
                xs = [x for f, x in centers.get(track_id, []) if seg["start_frame"] <= f <= seg["end_frame"]]
                expected = None

                if xs and (args.left or args.right):
                    expected = args.left if sum(xs) / len(xs) < width / 2 else args.right

                scores = {names.get(k, k): v for k, v in decision.scores.items()}
                verdict = ""

                if expected:
                    if got is None:
                        verdict, key = "→ 認不出", "認不出（沒上傳）"
                    elif got == expected:
                        verdict, key = "→ 正確", "認對"
                    else:
                        verdict, key = f"→ 認錯（實際是 {expected}）", "認錯"

                    outcome[key] += 1

                label = got or f"認不出（{decision.reason}）"
                print(f"  追蹤 {track_id} {seg['action']:<9}{seconds:5.1f}s 取樣{decision.samples:>3} → {label}  分數 {scores} {verdict}")

        if outcome:
            total = sum(outcome.values())
            print("\n統計（以左右位置為答案）：" + "、".join(f"{k} {v} 段（{v / total:.0%}）" for k, v in outcome.items()))
            print("最要緊的是「認錯」要接近 0；認不出只是少一些資料，不會讓統計出錯。")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
