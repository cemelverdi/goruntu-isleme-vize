"""
Bölüm 2 - Veri setinin ön incelemesi.

Üretilenler
-----------
results/data_frame_stats.csv      : 01 ve 02 sekanslarındaki her karenin yoğunluk istatistikleri
results/data_sequence_summary.csv : sekans bazında özet
results/devval_cell_stats.csv     : geliştirme+doğrulama karelerinde TRA işaretleyicilerinden
                                    hücre alanı ve hücre içi ortalama parlaklık
figures/intensity_over_time.png   : kare numarasına göre yoğunluk yüzdelikleri (veri görüntüsü içermez)
figures/histograms.png            : iki sekansın örnek histogramları (veri görüntüsü içermez)
figures/devval_cell_stats.png     : hücre alanı / parlaklık dağılımları (veri görüntüsü içermez)
report_figures/data_examples.png  : örnek kareler ve zorluk örnekleri (yalnızca rapor için)

Not: Sekans 02 (test) için yalnızca ham görüntü istatistikleri hesaplanır;
02_GT klasöründeki hiçbir dosya okunmaz.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops

from common import (DATA_ROOT, FIGURES, REPORT_FIGS, RESULTS, display_stretch,
                    load_selection, read_image, tra_path)

N_FRAMES = 92
PCTS = [1, 50, 90, 99, 99.9]


def frame_stats() -> pd.DataFrame:
    rows = []
    for seq in ("01", "02"):
        for t in range(N_FRAMES):
            im = read_image(seq, t)
            p = np.percentile(im, PCTS)
            rows.append({
                "sequence": seq, "frame_index": t,
                "height": im.shape[0], "width": im.shape[1], "dtype": str(im.dtype),
                "min": int(im.min()), "max": int(im.max()),
                "mean": float(im.mean()), "std": float(im.std()),
                **{f"p{q:g}": float(v) for q, v in zip(PCTS, p)},
                "frac_zero": float((im == 0).mean()),
                "n_unique_levels": int(np.unique(im).size),
            })
    return pd.DataFrame(rows)


def devval_cell_stats() -> pd.DataFrame:
    """Geliştirme ve doğrulama karelerinde (yalnızca sekans 01) hücre bazında ölçümler."""
    sel = load_selection(["development", "validation"])
    assert set(sel["sequence"]) == {"01"}, "Test referanslarına erişilmemeli"
    rows = []
    for _, r in sel.iterrows():
        img = read_image("01", int(r.frame_index))
        lab = tifffile.imread(tra_path("01", int(r.frame_index)))
        for rp in regionprops(lab, intensity_image=img):
            rows.append({
                "sample_id": r.sample_id, "group": r.group, "frame_index": r.frame_index,
                "cell_id": rp.label, "marker_area_px": rp.area,
                "mean_intensity": float(rp.intensity_mean),
                "equiv_diameter_px": float(rp.equivalent_diameter_area),
            })
    return pd.DataFrame(rows)


def plot_intensity_over_time(df: pd.DataFrame):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for a, seq in zip(ax, ("01", "02")):
        d = df[df.sequence == seq]
        for col in ("p99.9", "p99", "p90", "mean"):
            a.plot(d.frame_index, d[col], label=col)
        a.set_title(f"Sequence {seq}")
        a.set_xlabel("Kare numarası")
        a.grid(alpha=.3)
    ax[0].set_ylabel("Yoğunluk (uint8)")
    ax[1].legend(frameon=False)
    fig.suptitle("Kareler boyunca yoğunluk istatistikleri")
    fig.tight_layout()
    fig.savefig(FIGURES / "intensity_over_time.png", dpi=150)
    plt.close(fig)


def plot_histograms():
    """(a) 4'lük kutularla histogram, (b) ham seviyelerde tarak yapısı, (c) CDF."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.9))
    for seq, t, c in (("01", 1, "C0"), ("02", 23, "C1")):
        im = read_image(seq, t)
        lbl = f"Seq {seq}, t{t:03d} ({np.unique(im).size} seviye)"
        h4, edges = np.histogram(im, bins=64, range=(0, 256))
        ax[0].step(edges[:-1], h4 + 1, where="post", color=c, label=lbl)
        h = np.bincount(im.ravel(), minlength=256)
        ax[1].vlines(np.arange(256)[h > 0], 1, h[h > 0], color=c, alpha=.7, label=lbl)
        cdf = np.cumsum(h) / h.sum()
        ax[2].plot(np.arange(256), cdf, color=c, label=lbl)
    ax[0].set_yscale("log")
    ax[0].set_title("Histogram (4'lük kutular, log ölçek)")
    ax[0].set_ylabel("Piksel sayısı + 1")
    ax[1].set_yscale("log")
    ax[1].set_xlim(0, 80)
    ax[1].set_title("Ham gri seviyeler (0-80): boş seviyeler")
    ax[2].set_title("Kümülatif dağılım (0.95-1 aralığı)")
    ax[2].set_ylim(0.95, 1.0005)
    for a in ax:
        a.set_xlabel("Yoğunluk")
        a.legend(frameon=False, fontsize=8)
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "histograms.png", dpi=150)
    plt.close(fig)


def plot_cell_stats(cells: pd.DataFrame):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].hist(cells.marker_area_px, bins=30, color="C0")
    ax[0].set_title("TRA işaretleyici alanı (geliştirme+doğrulama)")
    ax[0].set_xlabel("Alan (piksel)")
    ax[1].hist(cells.mean_intensity, bins=30, color="C2")
    ax[1].set_title("Hücre içi ortalama yoğunluk")
    ax[1].set_xlabel("Ortalama yoğunluk (uint8)")
    for a in ax:
        a.set_ylabel("Hücre sayısı")
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "devval_cell_stats.png", dpi=150)
    plt.close(fig)


def plot_examples():
    """Rapor için örnek kareler ve zorluk örnekleri (veri görüntüsü içerir)."""
    im1 = read_image("01", 1)
    im2 = read_image("02", 23)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
    ax[0].imshow(display_stretch(im1), cmap="gray")
    ax[0].set_title("Seq 01, t001 (geliştirme) – görüntüleme için germe")
    ax[1].imshow(display_stretch(im2), cmap="gray")
    ax[1].set_title("Seq 02, t023 (test) – görüntüleme için germe")
    ax[2].imshow(im2, cmap="gray", vmin=0, vmax=255)
    ax[2].set_title("Seq 02, t023 – ham (0-255)")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(REPORT_FIGS / "data_examples.png", dpi=120)
    plt.close(fig)


def main():
    df = frame_stats()
    df.to_csv(RESULTS / "data_frame_stats.csv", index=False)

    summ = df.groupby("sequence").agg(
        n_frames=("frame_index", "size"),
        shape=("height", lambda s: f"{s.iloc[0]}x{df.width.iloc[0]}"),
        dtype=("dtype", "first"),
        max_intensity=("max", "max"),
        mean_mean=("mean", "mean"),
        p99_mean=("p99", "mean"),
        p99_9_mean=("p99.9", "mean"),
        frac_zero_mean=("frac_zero", "mean"),
        unique_levels_mean=("n_unique_levels", "mean"),
    ).round(3)
    summ.to_csv(RESULTS / "data_sequence_summary.csv")
    print("Sekans özeti:\n", summ.to_string(), "\n")

    cells = devval_cell_stats()
    cells.to_csv(RESULTS / "devval_cell_stats.csv", index=False)
    q = cells[["marker_area_px", "mean_intensity", "equiv_diameter_px"]].describe(
        percentiles=[.05, .5, .95]).round(1)
    print("Geliştirme+doğrulama hücre istatistikleri:\n", q.to_string(), "\n")
    print(f"En parlak / en sönük hücre ortalama yoğunluk oranı: "
          f"{cells.mean_intensity.max() / cells.mean_intensity.min():.1f}")

    plot_intensity_over_time(df)
    plot_histograms()
    plot_cell_stats(cells)
    plot_examples()
    print("Şekiller kaydedildi.")


if __name__ == "__main__":
    main()
