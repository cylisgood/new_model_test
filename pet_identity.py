"""
一支攝影機有多隻寵物時，判斷「每段行為是哪一隻寵物」，上傳時才能寫進對的 pet_id。

追蹤編號（Pet 1、Pet 2）不能直接當寵物身分：它重啟後會對調，而且動物重疊時會互換。
所以身分是「每個行為區段」依外觀重新判斷：

  1. 種類：用 ImageNet 分類器看裁切圖是狗還是貓（DeepLabCut 只輸出骨架，不分種類）。
     種類跟寵物資料（pets.species）不同的直接排除。
  2. 外觀：同種類有多隻時，拿 App 裡的大頭照（pets.avatar_url）當參考，
     比對區段裡取樣的畫面：分類器特徵的餘弦相似度 + HSV 顏色直方圖。
  3. 信心門檻：最像的要夠像，而且跟第二像的差距要夠大，否則回傳「認不出」，不上傳。
     寧可少一段，也不要記到錯的寵物身上。

啟用：環境變數 PET_IDENTIFY=on（預設關閉，維持舊行為：全部記在 PET_ID 這一隻）。
需要 SUPABASE_* 登入資訊才讀得到寵物清單和大頭照；PET_IDS 可只比對指定的幾隻（逗號分隔 uuid）。
門檻：PET_ID_MIN_SCORE、PET_ID_MIN_MARGIN、PET_ID_SAMPLE_EVERY（每幾幀取樣一次）。

重要：預設門檻是估的，還沒用真實多寵物影片校正過；用 identity_check.py 看實際分數再調。
"""

import os
import urllib.parse
import urllib.request
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# ImageNet-1k 的類別編號：151~268 是各種狗品種，281~285 是貓（虎斑、虎貓、波斯、暹羅、埃及貓）
DOG_CLASSES = range(151, 269)
CAT_CLASSES = range(281, 286)

MIN_ANIMAL_MASS = 0.15      # 狗＋貓的機率加起來低於這個，代表裁切圖看起來不像貓狗（例如框到別的東西）
SPECIES_MIN_SHARE = 0.70    # 一段行為裡，至少這個比例的取樣投給同一種類，才算種類確定
MIN_SAMPLES = 3             # 一段行為至少要有幾個取樣才敢判斷
MAX_SAMPLES_PER_TRACK = 3000
WEIGHT_EMBED = 0.6
WEIGHT_HIST = 0.4

DEFAULT_MIN_SCORE = 0.55
DEFAULT_MIN_MARGIN = 0.08
DEFAULT_SAMPLE_EVERY = 4

AVATAR_CENTER_RATIO = 0.7   # 大頭照只取中間這個比例，少一點背景
MAX_AVATAR_BYTES = 10 * 1024 * 1024
HIST_BINS = 16


def _log(message):
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", "replace").decode("ascii"))


# ============================================================
# 環境變數
# ============================================================

def read_identify_mode(env=None):
    """PET_IDENTIFY：沒設或 off 代表關閉；on 開啟。其他值報錯，避免打錯字卻以為有開。"""
    env = os.environ if env is None else env
    raw = env.get("PET_IDENTIFY", "").strip().lower()

    if raw in ("", "off", "0", "false"):
        return False

    if raw in ("on", "1", "true"):
        return True

    raise ValueError(f"PET_IDENTIFY 只支援 on、off，收到：{raw!r}")


def read_pet_ids(env=None):
    """PET_IDS：只比對這幾隻寵物（逗號分隔 uuid）。沒設回傳 None，代表帳號底下全部。"""
    env = os.environ if env is None else env
    raw = env.get("PET_IDS", "").strip()

    if not raw:
        return None

    return [str(uuid.UUID(part.strip())) for part in raw.split(",") if part.strip()]


def _read_float(env, name, default, minimum=0.0, maximum=1.0):
    raw = env.get(name, "").strip()

    if not raw:
        return default

    value = float(raw)

    if not minimum <= value <= maximum:
        raise ValueError(f"{name} 要在 {minimum} 到 {maximum} 之間")

    return value


def read_thresholds(env=None):
    """回傳 (最低分數, 最小差距, 取樣間隔幀數)。"""
    env = os.environ if env is None else env
    min_score = _read_float(env, "PET_ID_MIN_SCORE", DEFAULT_MIN_SCORE, -1.0, 1.0)
    min_margin = _read_float(env, "PET_ID_MIN_MARGIN", DEFAULT_MIN_MARGIN, 0.0, 2.0)
    every = int(_read_float(env, "PET_ID_SAMPLE_EVERY", DEFAULT_SAMPLE_EVERY, 1, 1000))
    return min_score, min_margin, every


# ============================================================
# 資料結構與純運算（不需要 torch，測試用得到）
# ============================================================

@dataclass
class Profile:
    pet_id: str
    name: str
    species: Optional[str] = None       # "dog" / "cat" / None（沒填）
    embed: Optional[np.ndarray] = field(default=None, repr=False)
    hist: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class Decision:
    pet_id: Optional[str]
    reason: str                          # 認得出時是 "ok"，否則是認不出的原因
    scores: dict = field(default_factory=dict)   # pet_id -> 分數，給 identity_check 和除錯看
    samples: int = 0


def _unit(vector):
    if vector is None:
        return None

    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else None


def hist_similarity(a, b):
    """HSV 直方圖相關係數，-1~1，越大越像。"""
    if a is None or b is None:
        return None

    return float(max(-1.0, min(1.0, cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_CORREL))))


def appearance_score(embed, hist, profile):
    """樣本（平均後）跟某個參考的外觀相似度；參考沒有外觀資料回傳 None。"""
    parts = []

    if embed is not None and profile.embed is not None:
        parts.append((WEIGHT_EMBED, float(np.dot(embed, profile.embed))))

    similarity = hist_similarity(hist, profile.hist)

    if similarity is not None:
        parts.append((WEIGHT_HIST, similarity))

    if not parts:
        return None

    total = sum(weight for weight, _ in parts)
    return sum(weight * value for weight, value in parts) / total


def aggregate_samples(samples):
    """
    一段行為裡所有取樣 → (種類, 種類占比, 平均特徵, 平均直方圖)。
    種類用信心加權投票；種類沒辦法判斷（None）的取樣不投票。
    """
    votes = {"dog": 0.0, "cat": 0.0}

    for sample in samples:
        if sample.get("species") in votes:
            votes[sample["species"]] += float(sample.get("species_conf", 1.0))

    total = sum(votes.values())
    species, share = None, 0.0

    if total > 0:
        species = max(votes, key=votes.get)
        share = votes[species] / total

    embeds = [s["embed"] for s in samples if s.get("embed") is not None]
    hists = [s["hist"] for s in samples if s.get("hist") is not None]
    embed = _unit(np.mean(embeds, axis=0)) if embeds else None
    hist = np.mean(hists, axis=0) if hists else None

    return species, share, embed, hist


def decide(samples, profiles, min_score=DEFAULT_MIN_SCORE, min_margin=DEFAULT_MIN_MARGIN):
    """
    這段行為是哪一隻？認不出就回傳 pet_id=None 並寫明原因。
    - 種類確定且跟寵物資料不同的，先排除；種類不確定（占比不夠）直接認不出
    - 剩下只有一隻：種類對上就算數；參考沒有種類資料時，要外觀分數過門檻
    - 剩下多隻：最高分過門檻，而且跟第二名差距夠大
    """
    n = len(samples)

    if not profiles:
        return Decision(None, "no_profiles", samples=n)

    if n < MIN_SAMPLES:
        return Decision(None, "too_few_samples", samples=n)

    species, share, embed, hist = aggregate_samples(samples)

    if species is None:
        return Decision(None, "species_unknown", samples=n)

    if share < SPECIES_MIN_SHARE:
        return Decision(None, "species_ambiguous", samples=n)

    candidates = [p for p in profiles if p.species in (None, species)]

    if not candidates:
        return Decision(None, "no_species_match", samples=n)

    scores = {p.pet_id: appearance_score(embed, hist, p) for p in candidates}
    shown = {pet_id: (None if s is None else round(s, 3)) for pet_id, s in scores.items()}

    if len(candidates) == 1:
        only = candidates[0]
        score = scores[only.pet_id]

        if only.species == species:
            return Decision(only.pet_id, "ok", shown, n)

        if score is not None and score >= min_score:
            return Decision(only.pet_id, "ok", shown, n)

        return Decision(None, "low_score", shown, n)

    if any(s is None for s in scores.values()):
        return Decision(None, "missing_reference", shown, n)

    ranked = sorted(candidates, key=lambda p: scores[p.pet_id], reverse=True)
    best, second = scores[ranked[0].pet_id], scores[ranked[1].pet_id]

    if best < min_score:
        return Decision(None, "low_score", shown, n)

    if best - second < min_margin:
        return Decision(None, "ambiguous", shown, n)

    return Decision(ranked[0].pet_id, "ok", shown, n)


# ============================================================
# 分類器（torchvision MobileNetV3，ImageNet 權重）：種類 + 特徵向量
# ============================================================

class CropEncoder:
    """
    第一次使用會從 download.pytorch.org 下載 MobileNetV3-Large 的 ImageNet 權重（約 21 MB），
    存在 torch 的快取資料夾，之後不用再下載。
    """

    def __init__(self, device=None):
        import torch
        import torchvision

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        self.model = torchvision.models.mobilenet_v3_large(weights=weights).eval().to(self.device)
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def encode_many(self, bgr_crops):
        """每張裁切圖回傳 (種類或 None, 種類信心, 單位長度特徵向量)。"""
        if not bgr_crops:
            return []

        torch = self._torch
        batch = np.stack([
            cv2.cvtColor(cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            for crop in bgr_crops
        ])
        x = torch.from_numpy(batch).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        x = (x - self._mean) / self._std

        with torch.no_grad():
            pooled = self.model.avgpool(self.model.features(x)).flatten(1)
            probs = torch.softmax(self.model.classifier(pooled), dim=1).cpu().numpy()
            embeds = pooled.cpu().numpy()

        results = []

        for p, e in zip(probs, embeds):
            dog = float(p[list(DOG_CLASSES)].sum())
            cat = float(p[list(CAT_CLASSES)].sum())

            if dog + cat < MIN_ANIMAL_MASS:
                species, conf = None, 0.0
            elif dog >= cat:
                species, conf = "dog", dog / (dog + cat)
            else:
                species, conf = "cat", cat / (dog + cat)

            results.append((species, conf, _unit(e)))

        return results


def crop_bbox(frame, bbox):
    if frame is None or bbox is None:
        return None

    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

    if x2 - x1 < 16 or y2 - y1 < 16:
        return None

    return frame[y1:y2, x1:x2]


def hsv_hist(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [HIST_BINS, HIST_BINS], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


# ============================================================
# 辨識器：逐幀收集取樣，行為區段結束時判斷是哪一隻
# ============================================================

class PetIdentifier:
    def __init__(self, profiles, encoder, min_score=DEFAULT_MIN_SCORE, min_margin=DEFAULT_MIN_MARGIN,
                 sample_every=DEFAULT_SAMPLE_EVERY):
        self.profiles = list(profiles)
        self.encoder = encoder
        self.min_score = min_score
        self.min_margin = min_margin
        self.sample_every = max(1, int(sample_every))
        self._samples = {}   # 追蹤編號 -> deque of 取樣（含 frame）

    def will_sample(self, frame_id):
        return frame_id % self.sample_every == 0

    def observe(self, track_id, frame_id, frame, bbox):
        """每幀每隻已確認的追蹤呼叫一次；只有每 sample_every 幀才真的取樣。"""
        if not self.will_sample(frame_id):
            return

        crop = crop_bbox(frame, bbox)

        if crop is None:
            return

        (species, conf, embed), = self.encoder.encode_many([crop])
        bucket = self._samples.setdefault(track_id, deque(maxlen=MAX_SAMPLES_PER_TRACK))
        bucket.append({
            "frame": frame_id,
            "species": species,
            "species_conf": conf,
            "embed": embed,
            "hist": hsv_hist(crop),
        })

    def resolve(self, track_id, segment):
        """行為區段 → 是哪一隻。用完就丟掉這段以前的取樣，記憶體不會一直長。"""
        bucket = self._samples.get(track_id)

        if not bucket:
            return Decision(None, "too_few_samples")

        start, end = segment["start_frame"], segment["end_frame"]
        inside = [s for s in bucket if start <= s["frame"] <= end]
        decision = decide(inside, self.profiles, self.min_score, self.min_margin)

        while bucket and bucket[0]["frame"] <= end:
            bucket.popleft()

        return decision


# ============================================================
# 從 Supabase 建立參考（寵物清單 + 大頭照）
# ============================================================

def _fetch_avatar(url, supabase_host, opener=None):
    """只允許從 Supabase 自己的網域抓大頭照（https），避免資料庫裡被塞別的網址。"""
    parsed = urllib.parse.urlparse(url or "")

    if parsed.scheme != "https" or parsed.hostname != supabase_host:
        raise ValueError("大頭照網址不是這個 Supabase 專案的 https 網址")

    opener = opener or urllib.request.urlopen

    with opener(url, timeout=15) as response:
        data = response.read(MAX_AVATAR_BYTES + 1)

    if len(data) > MAX_AVATAR_BYTES:
        raise ValueError("大頭照檔案太大")

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("大頭照無法解碼")

    return image


def build_profiles(pets, encoder, supabase_host, fetch=_fetch_avatar, log=_log):
    """
    pets：[{id, name, species, avatar_url}]。沒有種類也沒有可用大頭照的寵物沒有任何比對依據，略過並提示。
    大頭照的種類跟 pets.species 對不上時只印警告（以使用者填的 species 為準）。
    """
    profiles = []

    for pet in pets:
        name = pet.get("name") or pet.get("id")
        species = pet.get("species") if pet.get("species") in ("dog", "cat") else None
        embed = hist = None

        if pet.get("avatar_url"):
            try:
                image = fetch(pet["avatar_url"], supabase_host)
                h, w = image.shape[:2]
                dh, dw = int(h * (1 - AVATAR_CENTER_RATIO) / 2), int(w * (1 - AVATAR_CENTER_RATIO) / 2)
                center = image[dh:h - dh, dw:w - dw]
                (found, _conf, embed), = encoder.encode_many([center])
                hist = hsv_hist(center)

                if species and found and found != species:
                    log(f"⚠️ {name} 的大頭照看起來是 {found}，但寵物資料填的是 {species}，以寵物資料為準")
            except Exception as e:
                log(f"⚠️ {name} 的大頭照讀不到，外觀比對會缺這隻：{e}")

        if species is None and embed is None:
            log(f"⚠️ {name} 沒有種類也沒有大頭照，無法辨認，略過")
            continue

        profiles.append(Profile(pet["id"], name, species, embed, hist))

    return profiles


def build_identifier(client, supabase_settings, pet_ids=None, thresholds=None, encoder=None, log=_log):
    """登入、讀寵物清單、建立參考；回傳 PetIdentifier。client 是 behavior_export.SupabaseClient。"""
    client.sign_in()
    pets = client.list_pets_with_avatars()

    if pet_ids is not None:
        pets = [pet for pet in pets if pet.get("id") in pet_ids]

    if not pets:
        raise ValueError("找不到要辨認的寵物（檢查 PET_IDS，或這個帳號底下有沒有寵物）")

    encoder = encoder or CropEncoder()
    host = urllib.parse.urlparse(supabase_settings.url).hostname
    profiles = build_profiles(pets, encoder, host, log=log)

    if not profiles:
        raise ValueError("沒有任何寵物有種類或大頭照可以比對")

    min_score, min_margin, every = thresholds or read_thresholds()
    log("🐾 寵物辨認已啟用：" + "、".join(f"{p.name}（{p.species or '未填種類'}）" for p in profiles))
    return PetIdentifier(profiles, encoder, min_score, min_margin, every)


def describe(decision, profiles):
    """給 log 看的一行說明。"""
    names = {p.pet_id: p.name for p in profiles}

    if decision.pet_id:
        return f"{names.get(decision.pet_id, decision.pet_id)}"

    return f"認不出（{decision.reason}）"
