"""
Bölüm 3 - Geliştirme, doğrulama ve test görüntülerinin seçilmesi.

Seed kuralı: öğrenci numarasının son 5 hanesi + 2026
    Öğrenci no: 244329066  ->  29066 + 2026 = 31092

Kurallar (ödev metni, Bölüm 3):
  1. Sequence 01'den rastgele 5 doğrulama görüntüsü.
  2. Aynı sekanstan rastgele 10 geliştirme görüntüsü.
  3. Geliştirme ve doğrulama çakışmaz; aralarında en az 5 karelik boşluk.
     Yorum: her geliştirme karesi d ve her doğrulama karesi v için
     aradaki kare sayısı |d - v| - 1 >= 5  (yani |d - v| >= 6).
  4. Sequence 02'den rastgele 3 başlangıç noktası, her birinden ardışık 5 kare.
  5. Test blokları çakışmaz; aralarında en az 5 karelik boşluk
     (bir sonraki bloğun başı - önceki bloğun sonu - 1 >= 5).
  6. Koşullar sağlanamazsa aynı seed ile başlatılmış tek bir üreteçten
     deterministik olarak yeniden çekim yapılır (deneme numarası kaydedilir).
  7. Dosya adı, kare numarası, veri grubu ve seed selected_frames.csv'ye yazılır.

Not: Bu betik referans (GT) dosyalarının yalnızca VAR OLUP OLMADIĞINA bakar;
hiçbir etiket görüntüsünü okumaz. Böylece test referanslarına erişilmeden
seçim yapılmış olur.

Kullanım:
    python src/select_frames.py --data-root data/Fluo-N2DH-GOWT1 --out selected_frames.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
STUDENT_NUMBER = "244329066"
SEED = int(STUDENT_NUMBER[-5:]) + 2026  # 29066 + 2026 = 31092

N_VALIDATION = 5
N_DEVELOPMENT = 10
N_TEST_BLOCKS = 3
TEST_BLOCK_LEN = 5
MIN_GAP = 5            # iki grup / iki blok arasında en az bu kadar kare
MAX_ATTEMPTS = 1000    # deterministik tekrar sınırı

IMG_PATTERN = re.compile(r"^t(\d{3,4})\.tif$")


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------
def list_frames(data_root: Path, seq: str) -> dict[int, tuple[str, str]]:
    """Görüntüsü ve TRA anotasyon dosyası birlikte bulunan kareleri döndürür.

    Dönüş: {kare_no: (görüntü_yolu, tra_yolu)}  (yollar data_root'a göreli)
    """
    img_dir = data_root / seq
    tra_dir = data_root / f"{seq}_GT" / "TRA"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Görüntü klasörü bulunamadı: {img_dir}")
    if not tra_dir.is_dir():
        raise FileNotFoundError(f"TRA klasörü bulunamadı: {tra_dir}")

    frames: dict[int, tuple[str, str]] = {}
    for p in sorted(img_dir.iterdir()):
        m = IMG_PATTERN.match(p.name)
        if not m:
            continue
        idx_str = m.group(1)
        tra = tra_dir / f"man_track{idx_str}.tif"
        if tra.exists():
            frames[int(idx_str)] = (
                str(p.relative_to(data_root)),
                str(tra.relative_to(data_root)),
            )
    if not frames:
        raise RuntimeError(f"{seq} için görüntü+TRA çifti bulunamadı.")
    return frames


def frames_between(a: int, b: int) -> int:
    """a ve b kareleri arasında kalan kare sayısı."""
    return abs(a - b) - 1


def select_dev_val(frames: list[int], rng: np.random.Generator):
    """Önce 5 doğrulama, sonra boşluk kuralına uyan 10 geliştirme karesi."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        val = np.sort(rng.choice(frames, size=N_VALIDATION, replace=False))
        eligible = [
            f for f in frames
            if all(frames_between(f, v) >= MIN_GAP for v in val)
        ]
        if len(eligible) < N_DEVELOPMENT:
            continue  # koşul sağlanamadı -> aynı üreteçle yeniden çek
        dev = np.sort(rng.choice(eligible, size=N_DEVELOPMENT, replace=False))
        return val.tolist(), dev.tolist(), attempt
    raise RuntimeError("Geliştirme/doğrulama seçimi MAX_ATTEMPTS içinde başarısız.")


def select_test_blocks(frames: list[int], rng: np.random.Generator):
    """3 adet çakışmayan, aralarında >= 5 kare boşluk olan 5'li ardışık blok."""
    frame_set = set(frames)
    # Bloğun 5 karesinin tamamı mevcut olan başlangıç noktaları
    starts = [
        s for s in frames
        if all((s + k) in frame_set for k in range(TEST_BLOCK_LEN))
    ]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        chosen = np.sort(rng.choice(starts, size=N_TEST_BLOCKS, replace=False))
        ok = all(
            chosen[i + 1] - (chosen[i] + TEST_BLOCK_LEN - 1) - 1 >= MIN_GAP
            for i in range(N_TEST_BLOCKS - 1)
        )
        if ok:
            blocks = [
                [int(s) + k for k in range(TEST_BLOCK_LEN)] for s in chosen
            ]
            return blocks, attempt
    raise RuntimeError("Test bloğu seçimi MAX_ATTEMPTS içinde başarısız.")


def check_rules(val, dev, blocks):
    """Seçimin bütün kurallara uyduğunu doğrular (assert)."""
    assert len(val) == N_VALIDATION and len(set(val)) == N_VALIDATION
    assert len(dev) == N_DEVELOPMENT and len(set(dev)) == N_DEVELOPMENT
    assert not set(val) & set(dev), "Geliştirme ve doğrulama çakışıyor"
    assert all(frames_between(d, v) >= MIN_GAP for d in dev for v in val), \
        "Geliştirme-doğrulama boşluk kuralı ihlal edildi"
    assert len(blocks) == N_TEST_BLOCKS
    for b in blocks:
        assert len(b) == TEST_BLOCK_LEN and b == list(range(b[0], b[0] + TEST_BLOCK_LEN))
    for b1, b2 in zip(blocks, blocks[1:]):
        assert frames_between(b2[0], b1[-1]) >= MIN_GAP, "Test blokları arası boşluk yetersiz"


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------
def select_frames(data_root: Path) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)  # tek üreteç: tüm seçim bu sırayla yapılır

    seq01 = list_frames(data_root, "01")
    seq02 = list_frames(data_root, "02")

    val, dev, att_dv = select_dev_val(sorted(seq01), rng)
    blocks, att_t = select_test_blocks(sorted(seq02), rng)
    check_rules(val, dev, blocks)

    rows = []
    for f in val:
        rows.append(("validation", "01", f, *seq01[f], "", att_dv))
    for f in dev:
        rows.append(("development", "01", f, *seq01[f], "", att_dv))
    for bi, block in enumerate(blocks, start=1):
        for f in block:
            rows.append(("test", "02", f, *seq02[f], f"T{bi}", att_t))

    df = pd.DataFrame(rows, columns=[
        "group", "sequence", "frame_index", "image_file", "tra_file",
        "test_block", "selection_attempt",
    ])
    order = {"development": 0, "validation": 1, "test": 2}
    df = df.sort_values(["group", "frame_index"], key=lambda c: c.map(order) if c.name == "group" else c)
    df.insert(0, "sample_id", [f"{g[0].upper()}{i:02d}" for g, i in
                               zip(df["group"], df.groupby("group").cumcount() + 1)])
    df["seed"] = SEED
    df.attrs["n_frames_01"] = len(seq01)
    df.attrs["n_frames_02"] = len(seq02)
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Seed'e dayalı tekrarlanabilir kare seçimi")
    ap.add_argument("--data-root", default="data/Fluo-N2DH-GOWT1", type=Path)
    ap.add_argument("--out", default="selected_frames.csv", type=Path)
    args = ap.parse_args()

    df = select_frames(args.data_root)
    df.to_csv(args.out, index=False)

    print(f"Seed: {SEED}  (= {STUDENT_NUMBER[-5:]} + 2026)")
    print(f"Sequence 01 kare sayısı: {df.attrs['n_frames_01']}, "
          f"Sequence 02 kare sayısı: {df.attrs['n_frames_02']}")
    for g in ["development", "validation", "test"]:
        sub = df[df.group == g]
        print(f"{g:12s} ({len(sub):2d}): {sub.frame_index.tolist()}")
    print(f"Kaydedildi -> {args.out}")


if __name__ == "__main__":
    main()
