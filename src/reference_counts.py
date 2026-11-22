"""
Bölüm 4 - Referans hücre sayılarının belirlenmesi.

Yöntem: ilgili karenin TRA etiket görüntüsü okunur, 0 (arka plan) hariç
birbirinden farklı pozitif hücre kimlikleri sayılır.

GÜVENLİK: Varsayılan olarak YALNIZCA geliştirme ve doğrulama karelerinin
referansları üretilir. Test referansları ancak test tahminleri
test_predictions_locked.csv dosyasına kaydedildikten sonra
`--include-test` bayrağıyla okunabilir (Bölüm 12). Bayrak verildiğinde
kilitli tahmin dosyası yoksa betik hata verir.

Çıktılar
--------
results/reference_counts_devval.csv   (geliştirme + doğrulama)
results/reference_counts_test.csv     (yalnızca --include-test ile)
report_figures/reference_check_*.png  (görsel kontrol, yalnızca rapor)
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries

from common import (REPORT_FIGS, RESULTS, ROOT, display_stretch, load_selection,
                    read_image, tra_path)

LOCKED_PREDICTIONS = ROOT / "test_predictions_locked.csv"


def count_reference(seq: str, t: int) -> dict:
    lab = tifffile.imread(tra_path(seq, t))
    ids = np.unique(lab)
    ids = ids[ids > 0]
    # Ek kontrol: aynı kimlik birden fazla parçaya bölünmüş mü?
    n_fragments = int(sum(label(lab == i).max() for i in ids))
    return {"reference_count": int(ids.size),
            "n_marker_fragments": n_fragments,
            "max_label": int(lab.max())}


def visual_check(rows: pd.DataFrame, fname: str):
    n = len(rows)
    cols = 5
    nrow = int(np.ceil(n / cols))
    fig, axes = plt.subplots(nrow, cols, figsize=(4 * cols, 4.2 * nrow))
    for ax, (_, r) in zip(axes.ravel(), rows.iterrows()):
        img = read_image(r.sequence, int(r.frame_index))
        lab = tifffile.imread(tra_path(r.sequence, int(r.frame_index)))
        rgb = np.dstack([display_stretch(img)] * 3)
        rgb[find_boundaries(lab, mode="outer")] = (1, 0.2, 0.2)
        ax.imshow(rgb)
        for rp in regionprops(lab):
            y, x = rp.centroid
            ax.text(x, y, str(rp.label), color="yellow", fontsize=6,
                    ha="center", va="center")
        ax.set_title(f"{r.sample_id} – t{int(r.frame_index):03d} – n={r.reference_count}",
                     fontsize=10)
        ax.axis("off")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(REPORT_FIGS / fname, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-test", action="store_true",
                    help="Test referanslarını oku (yalnızca kilitli tahminlerden SONRA)")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    groups = ["development", "validation"]
    if args.include_test:
        if not LOCKED_PREDICTIONS.exists():
            raise SystemExit("HATA: test_predictions_locked.csv yok. Test referansları "
                             "ancak test tahminleri kilitlendikten sonra okunabilir.")
        groups = ["test"]

    sel = load_selection(groups)
    rows = [{**r.to_dict(), **count_reference(r.sequence, int(r.frame_index))}
            for _, r in sel.iterrows()]
    df = pd.DataFrame(rows)[["sample_id", "group", "sequence", "frame_index",
                             "test_block", "reference_count", "n_marker_fragments",
                             "max_label"]]
    df["fragmented"] = df.n_marker_fragments != df.reference_count

    out = RESULTS / ("reference_counts_test.csv" if args.include_test
                     else "reference_counts_devval.csv")
    df.to_csv(out, index=False)
    print(df.drop(columns=["test_block"] if not args.include_test else []).to_string(index=False))
    print("\nGrup bazında:\n", df.groupby("group").reference_count.describe().round(2).to_string())
    print(f"\nKaydedildi -> {out}")

    if not args.no_figures and not args.include_test:
        for g in groups:
            visual_check(df[df.group == g].reset_index(drop=True), f"reference_check_{g}.png")


if __name__ == "__main__":
    main()
