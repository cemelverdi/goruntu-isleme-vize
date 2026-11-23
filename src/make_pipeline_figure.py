"""Şekil 4 - Nihai pipeline'ın adım adım çıktıları (doğrulama karesi, yalnızca rapor)."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb
from skimage.segmentation import find_boundaries

from common import REPORT_FIGS, ROOT, display_stretch, read_image
from evaluate import references
from pipeline import run_pipeline

SAMPLE = "V01"


def main():
    final = json.loads((ROOT / "final_pipeline.json").read_text())
    cfg = final["pipeline"]
    ref = references("validation").set_index("sample_id").loc[SAMPLE]
    t = int(ref.frame_index)
    img = read_image("01", t)
    o = run_pipeline(img, cfg, return_steps=True)
    panels = [
        ("1) Orijinal (görüntüleme için germe)", display_stretch(img), "gray"),
        ("2) Log dönüşümü", o["intensity"], "gray"),
        ("3) Gauss 5x5 + Laplace keskinleştirme", o["filtered"], "gray"),
        (f"4) Otsu eşikleme (T={o['threshold']:.0f})", o["mask_raw"], "gray"),
        ("5) Closing r=3 → opening r=3 → delik doldurma", o["mask"], "gray"),
    ]
    fig, ax = plt.subplots(2, 3, figsize=(15, 10.4))
    for a, (title, im, cmap) in zip(ax.ravel(), panels):
        a.imshow(im, cmap=cmap)
        a.set_title(title, fontsize=10)
    rgb = label2rgb(o["labels"], image=display_stretch(img), bg_label=0, alpha=.35)
    rgb[find_boundaries(o["labels"], mode="outer")] = (1, 1, 0)
    ax[1, 2].imshow(rgb)
    ax[1, 2].set_title(f"6) Watershed (min_dist=15) + min alan 400 px\n"
                       f"tahmin = {o['count']}, referans = {int(ref.reference_count)}", fontsize=10)
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle(f"Şekil 4. Nihai pipeline'ın adım adım çıktıları – {SAMPLE} (t{t:03d}, doğrulama)",
                 fontsize=12)
    fig.tight_layout(h_pad=3)
    fig.savefig(REPORT_FIGS / "fig4_final_pipeline_steps.png", dpi=110)
    plt.close(fig)
    print("Şekil 4 kaydedildi.")


if __name__ == "__main__":
    main()
