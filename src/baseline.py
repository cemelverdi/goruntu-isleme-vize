"""
Bölüm 5 - Başlangıç yöntemi (baseline).

Adımlar
  1. Gri seviyeli görüntünün okunması (ön işleme yok)
  2. Otsu eşikleme (global, görüntü başına otomatik eşik)
  3. Bağlı bileşen analizi (8-komşuluk)
  4. Nesnelerin sayılması
  5. Segmentasyon çıktısının ve nesnelerin görselleştirilmesi

İzin verilen tek ek: geliştirme verisi üzerinde belirlenen SABİT bir minimum
nesne alanı eşiği. Aday değerler yalnızca 10 geliştirme karesinde denenir,
en düşük MAE'yi veren (eşitlikte en küçük) değer seçilir ve
results/baseline_config.json dosyasına yazılarak sabitlenir.
Bu dosya oluştuktan sonra baseline bir daha ayarlanmaz.

Kullanım
  python src/baseline.py --tune        # min alanı geliştirme setinde belirle ve sabitle
  python src/baseline.py               # sabit baseline'ı geliştirme + doğrulama karelerinde çalıştır
"""

from __future__ import annotations

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.color import label2rgb
from skimage.filters import threshold_otsu
from skimage.measure import label

from common import REPORT_FIGS, RESULTS, display_stretch, load_selection, read_image
import tifffile

from common import tra_path
from metrics import count_metrics, error_decomposition

CONFIG_PATH = RESULTS / "baseline_config.json"
MIN_AREA_CANDIDATES = [0, 25, 50, 100, 200, 400]
CONNECTIVITY = 2  # 8-komşuluk


def filter_small(labels: np.ndarray, min_area: int):
    """Alanı min_area'dan KÜÇÜK bileşenleri siler (alan >= min_area kalır).
    scikit-image sürümünden bağımsız, açık tanımlı filtre."""
    areas = np.bincount(labels.ravel())
    keep = areas >= min_area
    keep[0] = False
    mask = keep[labels]
    return label(mask, connectivity=CONNECTIVITY), mask


def run_baseline(img: np.ndarray, min_area: int) -> dict:
    """Baseline'ı tek görüntüye uygular. Ara çıktıları da döndürür."""
    t0 = time.perf_counter()
    thr = threshold_otsu(img)
    mask = img > thr
    labels = label(mask, connectivity=CONNECTIVITY)
    if min_area > 0:
        labels, mask = filter_small(labels, min_area)
    count = int(labels.max())
    elapsed = time.perf_counter() - t0
    return {"threshold": float(thr), "mask": mask, "labels": labels,
            "count": count, "time_s": elapsed}


def tune_min_area() -> int:
    if CONFIG_PATH.exists():
        raise SystemExit(f"Baseline zaten sabitlenmiş: {CONFIG_PATH}. Yeniden ayarlama yapılmaz.")
    ref = pd.read_csv(RESULTS / "reference_counts_devval.csv", dtype={"sequence": str})
    dev = ref[ref.group == "development"]
    imgs = {int(r.frame_index): read_image("01", int(r.frame_index)) for _, r in dev.iterrows()}

    rows = []
    for a in MIN_AREA_CANDIDATES:
        preds = [run_baseline(imgs[int(t)], a)["count"] for t in dev.frame_index]
        m = count_metrics(dev.reference_count, preds)
        rows.append({"min_area": a, **m, "predictions": preds})
    tab = pd.DataFrame(rows)
    tab.to_csv(RESULTS / "baseline_min_area_tuning_dev.csv", index=False)
    print(tab.to_string(index=False))

    best = tab.sort_values(["MAE", "min_area"]).iloc[0]
    cfg = {
        "method": "baseline",
        "steps": ["read_gray", "otsu_threshold", "remove_small_objects", "connected_components"],
        "connectivity": "8",
        "min_area": int(best.min_area),
        "selection_rule": "min development MAE over candidates; tie -> smaller min_area",
        "candidates": MIN_AREA_CANDIDATES,
        "dev_MAE": float(best.MAE),
    }
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    print(f"\nSeçilen min_area = {cfg['min_area']}  ->  {CONFIG_PATH}")
    return cfg["min_area"]


def visualize(examples: list[tuple[str, np.ndarray, dict, int]], fname: str):
    fig, axes = plt.subplots(len(examples), 3, figsize=(13.5, 4.6 * len(examples)))
    axes = np.atleast_2d(axes)
    for row, (title, img, out, ref) in zip(axes, examples):
        row[0].imshow(display_stretch(img), cmap="gray")
        row[0].set_title(f"{title} – görüntü (görüntüleme için germe)")
        row[1].imshow(out["mask"], cmap="gray")
        row[1].set_title(f"Otsu maske (T={out['threshold']:.0f})")
        row[2].imshow(label2rgb(out["labels"], image=display_stretch(img), bg_label=0, alpha=.45))
        row[2].set_title(f"Bağlı bileşenler: tahmin={out['count']}, referans={ref}")
        for a in row:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(REPORT_FIGS / fname, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    args = ap.parse_args()
    if args.tune:
        tune_min_area()

    cfg = json.loads(CONFIG_PATH.read_text())
    ref = pd.read_csv(RESULTS / "reference_counts_devval.csv", dtype={"sequence": str})

    rows, examples = [], []
    for _, r in ref.iterrows():
        img = read_image(r.sequence, int(r.frame_index))
        out = run_baseline(img, cfg["min_area"])
        rows.append({"sample_id": r.sample_id, "group": r.group, "frame_index": r.frame_index,
                     "reference_count": r.reference_count, "baseline_count": out["count"],
                     "error": out["count"] - r.reference_count,
                     "otsu_threshold": out["threshold"], "time_s": out["time_s"],
                     **error_decomposition(out["labels"],
                                           tifffile.imread(tra_path(r.sequence, int(r.frame_index))))})
        if r.sample_id in ("D01", "D10", "V01"):
            examples.append((f"{r.sample_id} t{int(r.frame_index):03d}", img, out,
                             int(r.reference_count)))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "baseline_devval.csv", index=False)
    print(df.round(4).to_string(index=False))

    summ = pd.DataFrame([{"group": g, **count_metrics(d.reference_count, d.baseline_count),
                          "mean_time_s": d.time_s.mean()} for g, d in df.groupby("group")])
    summ.to_csv(RESULTS / "baseline_devval_summary.csv", index=False)
    print("\n", summ.round(4).to_string(index=False))
    visualize(examples, "baseline_examples.png")


if __name__ == "__main__":
    main()
