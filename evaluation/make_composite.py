"""
拼接測試影片產生器

把兩支「各只有一隻寵物」的影片拼成一支「兩隻寵物同時入鏡」的影片，
同時輸出 ground truth（每個時間點、畫面左 / 右半邊分別是哪一隻），
讓 evaluate_tracking.py 不用人工標註就能自動算出追蹤 / 身分指標。

模式：
  side_by_side  兩隻固定在左右兩側（驗證：多隻偵測、追蹤、各自的行為輸出）
  swap          播到一半，兩隻的左右位置對調
                （驗證：編號是跟著「動物外觀」走，還是跟著「畫面位置」走）

限制（請務必知道）：
  兩隻動物各自在自己的畫面裡活動，永遠不會互相經過、遮擋、靠近，
  所以這個測試「不能」代表真實的多寵物情況，只能驗證流程與外觀區分能力。
  真實的遮擋 / 交錯要用實拍的兩隻同時入鏡影片。

用法（在專案根目錄）：
  venv\\Scripts\\python.exe evaluation\\make_composite.py
  venv\\Scripts\\python.exe evaluation\\make_composite.py --mode swap
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_DIR = Path(__file__).resolve().parent.parent
DEFAULT_LEFT = REPO_DIR / "input" / "14877397_1080_2320_30fps.mp4"   # 奶油色長毛貓（直式）
DEFAULT_RIGHT = REPO_DIR / "input" / "12756589_3840_2160_60fps.mp4"  # 橘虎斑貓（橫式 4K）
MAX_DEFAULT_SECONDS = 30.0


def edge_color(frame):
    """
    取畫面四周邊緣像素的中位數顏色，當作留白用的背景色。
    刻意不用「模糊放大的複本」當背景，避免偵測器把它當成第二隻動物。
    """
    h, w = frame.shape[:2]
    m = max(2, min(h, w) // 40)
    border = np.concatenate([
        frame[:m].reshape(-1, 3),
        frame[-m:].reshape(-1, 3),
        frame[:, :m].reshape(-1, 3),
        frame[:, -m:].reshape(-1, 3),
    ])
    return tuple(int(v) for v in np.median(border, axis=0))


def fit_to_panel(frame, bg_color, panel_w, panel_h):
    """等比例縮放放進固定大小的面板（不裁切，避免動物被切掉），剩下的空間用背景色補滿。"""
    fh, fw = frame.shape[:2]
    scale = min(panel_w / fw, panel_h / fh)
    new_w, new_h = int(round(fw * scale)), int(round(fh * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    panel = np.empty((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = bg_color
    x0, y0 = (panel_w - new_w) // 2, (panel_h - new_h) // 2
    panel[y0:y0 + new_h, x0:x0 + new_w] = resized
    return panel


class FrameSource:
    """依序讀一支影片，並依「輸出 fps」換算要取哪一幀（例如 60 fps 來源要降成 30 fps）。"""

    def __init__(self, path, start_seconds, out_fps):
        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            sys.exit(f"無法開啟影片：{self.path}")

        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or out_fps
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.ratio = self.src_fps / out_fps
        self.start_idx = int(round(start_seconds * self.src_fps))

        if self.start_idx >= self.total:
            sys.exit(f"{self.path.name} 只有 {self.total / self.src_fps:.1f} 秒，起點 {start_seconds} 秒超出範圍")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_idx)
        ok, first = self.cap.read()
        if not ok:
            sys.exit(f"無法從 {self.path.name} 讀取第一幀")

        self.pos = self.start_idx + 1  # 下一個會被讀到的來源幀編號
        self.last = first
        self.bg = edge_color(first)

    def max_output_frames(self):
        """從起點開始，最多能取出幾個輸出幀（最後一個要取的來源幀不能超過影片結尾）。"""
        return int((self.total - 1 - self.start_idx) / self.ratio) + 1

    def frame_at(self, k):
        target = self.start_idx + int(round(k * self.ratio))

        if target < self.pos:
            return self.last  # 來源 fps 比輸出低時，同一幀會重複使用

        while self.pos < target:
            if not self.cap.grab():
                return self.last
            self.pos += 1

        ok, frame = self.cap.read()
        if not ok:
            return self.last

        self.pos += 1
        self.last = frame
        return frame

    def release(self):
        self.cap.release()


def main():
    ap = argparse.ArgumentParser(description="把兩支單隻寵物的影片拼成兩隻同時入鏡的測試影片 + ground truth")
    ap.add_argument("--left", default=str(DEFAULT_LEFT), help="身分 A（一開始在左邊）的來源影片")
    ap.add_argument("--right", default=str(DEFAULT_RIGHT), help="身分 B（一開始在右邊）的來源影片")
    ap.add_argument("--left-start", type=float, default=24.0,
                    help="A 影片從第幾秒開始取（預設 24：這支貓在 27~31 秒有坐→站→走的動作變化）")
    ap.add_argument("--right-start", type=float, default=0.0, help="B 影片從第幾秒開始取")
    ap.add_argument("--seconds", type=float, default=None, help="輸出長度（秒），預設取兩支都夠長的最大值")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--panel-width", type=int, default=960)
    ap.add_argument("--panel-height", type=int, default=720)
    ap.add_argument("--mode", choices=["side_by_side", "swap"], default="side_by_side")
    ap.add_argument("--swap-at", type=float, default=None, help="swap 模式：第幾秒對調左右（預設一半）")
    ap.add_argument("--out", default=None, help="輸出路徑，預設 input\\composite_<mode>.mp4")
    args = ap.parse_args()

    left = FrameSource(args.left, args.left_start, args.fps)
    right = FrameSource(args.right, args.right_start, args.fps)

    n_max = min(left.max_output_frames(), right.max_output_frames())
    if args.seconds is None:
        n_out = min(n_max, int(MAX_DEFAULT_SECONDS * args.fps))
    else:
        n_out = int(args.seconds * args.fps)
        if n_out > n_max:
            sys.exit(f"要求 {args.seconds} 秒，但兩支影片最多只能湊出 {n_max / args.fps:.1f} 秒")

    if n_out < 1:
        sys.exit("輸出長度不足 1 幀")

    swap_frame = None
    if args.mode == "swap":
        swap_frame = int(round(args.swap_at * args.fps)) if args.swap_at is not None else n_out // 2
        if not 0 < swap_frame < n_out:
            sys.exit("--swap-at 必須落在影片長度之內")

    out_path = Path(args.out) if args.out else REPO_DIR / "input" / f"composite_{args.mode}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pw, ph = args.panel_width, args.panel_height
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (pw * 2, ph))
    if not writer.isOpened():
        sys.exit(f"無法寫入：{out_path}")

    for k in range(n_out):
        panel_a = fit_to_panel(left.frame_at(k), left.bg, pw, ph)
        panel_b = fit_to_panel(right.frame_at(k), right.bg, pw, ph)
        swapped = swap_frame is not None and k >= swap_frame
        writer.write(np.hstack([panel_b, panel_a] if swapped else [panel_a, panel_b]))

    writer.release()
    left.release()
    right.release()

    if swap_frame is None:
        segments = [{"start_frame": 0, "end_frame": n_out - 1, "left": "A", "right": "B"}]
    else:
        segments = [
            {"start_frame": 0, "end_frame": swap_frame - 1, "left": "A", "right": "B"},
            {"start_frame": swap_frame, "end_frame": n_out - 1, "left": "B", "right": "A"},
        ]

    gt = {
        "version": 1,
        "mode": args.mode,
        "video": out_path.name,
        "fps": args.fps,
        "frame_count": n_out,
        "width": pw * 2,
        "height": ph,
        "panel_width": pw,
        "identities": {
            "A": {"source": str(Path(args.left).name), "start_seconds": args.left_start},
            "B": {"source": str(Path(args.right).name), "start_seconds": args.right_start},
        },
        "segments": segments,
    }

    gt_path = out_path.with_suffix(".gt.json")
    gt_path.write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"影片：{out_path}（{n_out} 幀，{n_out / args.fps:.1f} 秒，{pw * 2}x{ph}，模式 {args.mode}）")
    print(f"Ground truth：{gt_path}")
    if swap_frame is not None:
        print(f"左右對調發生在第 {swap_frame} 幀（{swap_frame / args.fps:.1f} 秒）")


if __name__ == "__main__":
    main()
