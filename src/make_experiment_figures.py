"""
Deney şekilleri (Bölüm 17).

report_figures/ (veri görüntüsü içerir, yalnızca rapor):
  fig2_intensity_transforms.png   Şekil 2: yoğunluk dönüşümleri + histogramlar + Otsu maskeleri
  fig3_segmentation_morphology.png Şekil 3: segmentasyon ve morfoloji çıktılarının karşılaştırması

figures/ (yalnızca grafik, repoya girer):
  param_intensity.png      yoğunluk dönüşümlerinin MAE / hata bileşenleri
  param_filter.png         filtre ve kernel boyutunun etkisi
  param_threshold.png      sabit eşik ve adaptif eşik parametrelerinin etkisi
  param_morphology.png     morfolojik işlem ve sırasının etkisi
  param_minarea_ws.png     minimum alan ve watershed parametresi
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.color import label2rgb

from common import FIGURES, REPORT_FIGS, RESULTS, ROOT, read_image
from evaluate import hm_ref
from pipeline import make_config, run_pipeline

EX_FRAME = ("01", 1)                       # D01 (geliştirme)
CROP = (slice(330, 842), slice(360, 872))  # sönük + parlak hücrelerin bulunduğu 512x512 bölge
FH = {"op": "fill_holes"}

plt.rcParams.update({"font.size": 9})


def crop(a):
    return a[CROP]


def load_exp():
    # "(tekrar)" satırları, aşamanın bağlam (kontrol) ayarını gösterdiği için tutulur
    return pd.read_csv(ROOT / "experiments.csv")


# ---------------------------------------------------------------------------
# Şekil 2
# ---------------------------------------------------------------------------
FIG2_FRAME = ("01", 21, "D05")  # histogram matching referansı (D01) ile aynı kare olmasın


def fig2():
    img = read_image(*FIG2_FRAME[:2])
    methods = [("Orijinal", {"method": "none"}),
               ("Contrast stretching (0-99.5%)", {"method": "stretch", "p_low": 0, "p_high": 99.5}),
               ("Histogram eşitleme", {"method": "histeq"}),
               ("CLAHE (clip=4, 8x8)", {"method": "clahe", "clip": 4.0, "tiles": 8}),
               ("Histogram matching (ref: D01)", {"method": "histmatch"}),
               ("Log dönüşümü", {"method": "log"})]
    ref_counts = pd.read_csv(RESULTS / "reference_counts_devval.csv")
    ref_n = int(ref_counts.query(f"sample_id=='{FIG2_FRAME[2]}'").reference_count.iloc[0])
    fig, ax = plt.subplots(3, len(methods), figsize=(3.1 * len(methods), 9.6))
    for j, (name, icfg) in enumerate(methods):
        cfg = make_config(intensity=icfg, filter={"method": "gaussian", "ksize": 5},
                          morphology=[FH], min_area=200)
        out = run_pipeline(img, cfg, ref_img=hm_ref(), return_steps=True)
        x = out["intensity"]
        ax[0, j].imshow(crop(x), cmap="gray", vmin=0, vmax=1)
        ax[0, j].set_title(name, fontsize=9)
        u8 = np.round(x * 255).astype(np.uint8)
        h, edges = np.histogram(u8, bins=64, range=(0, 256))
        ax[1, j].bar(edges[:-1], h + 1, width=4, color="C0", align="edge")
        ax[1, j].set_yscale("log")
        ax[1, j].set_xlim(0, 256)
        if out["threshold"] is not None:
            ax[1, j].axvline(out["threshold"], color="C3", lw=1.2, label=f"Otsu T={out['threshold']:.0f}")
            ax[1, j].legend(fontsize=7, frameon=False)
        ax[1, j].set_xlabel("Yoğunluk (0-255)")
        ax[2, j].imshow(label2rgb(crop(out["labels"]), image=crop(x), bg_label=0, alpha=.4))
        ax[2, j].set_title(f"Gauss5 + Otsu + doldurma: n={out['count']} (ref={ref_n})", fontsize=8)
        for a in (ax[0, j], ax[2, j]):
            a.axis("off")
    ax[1, 0].set_ylabel("Piksel sayısı + 1 (log)")
    fig.suptitle(f"Şekil 2. Yoğunluk dönüşümleri ({FIG2_FRAME[2]}, t{FIG2_FRAME[1]:03d}; 512x512 kesit) – görüntü, histogram "
                 "ve aynı bağlamda segmentasyon (sayılar tüm kare içindir)", fontsize=10)
    fig.tight_layout()
    fig.savefig(REPORT_FIGS / "fig2_intensity_transforms.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Şekil 3
# ---------------------------------------------------------------------------
def fig3():
    img = read_image(*EX_FRAME)
    cand = json.loads((RESULTS / "candidate_pipelines.json").read_text())
    A = cand["A"]["config"]
    seg_variants = [
        ("Otsu (aday A bağlamı)", A["segmentation"]),
        ("Sabit eşik T=80 (log ölçeği)", {"method": "fixed", "threshold": 80}),
        ("Sabit eşik T=140 (log ölçeği)", {"method": "fixed", "threshold": 140}),
        ("Adaptif blok=101 C=20", {"method": "adaptive", "block": 101, "C": 20}),
        ("Üçgen (triangle)", {"method": "triangle"}),
        ("K-means k=3", {"method": "kmeans", "k": 3, "bg_clusters": 1}),
    ]
    morph_variants = [
        ("Morfoloji yok", []),
        ("Delik doldurma", [FH]),
        ("Erosion r=3 + doldurma", [{"op": "erode", "size": 3}, FH]),
        ("Dilation r=3 + doldurma", [{"op": "dilate", "size": 3}, FH]),
        ("Closing r=3 → opening r=3 + doldurma", A["morphology"]),
        ("Opening r=3 → closing r=3 + doldurma",
         [{"op": "open", "size": 3}, {"op": "close", "size": 3}, FH]),
    ]
    fig, ax = plt.subplots(2, 6, figsize=(19, 7.4))
    base = {k: A[k] for k in ("intensity", "filter")}
    for j, (name, seg) in enumerate(seg_variants):
        cfg = make_config(**base, segmentation=seg, morphology=[FH],
                          separation={"method": "none"}, min_area=400)
        out = run_pipeline(img, cfg, return_steps=True)
        ax[0, j].imshow(label2rgb(crop(out["labels"]), image=crop(out["intensity"]), bg_label=0, alpha=.45))
        ax[0, j].set_title(f"{name}\nn={out['count']}", fontsize=8.5)
    for j, (name, morph) in enumerate(morph_variants):
        cfg = make_config(**base, segmentation=A["segmentation"], morphology=morph,
                          separation={"method": "none"}, min_area=400)
        out = run_pipeline(img, cfg, return_steps=True)
        ax[1, j].imshow(label2rgb(crop(out["labels"]), image=crop(out["intensity"]), bg_label=0, alpha=.45))
        ax[1, j].set_title(f"{name}\nn={out['count']}", fontsize=8.5)
    for a in ax.ravel():
        a.axis("off")
    ax[0, 0].text(-30, 256, "Segmentasyon", rotation=90, va="center", ha="right", fontsize=11)
    ax[1, 0].text(-30, 256, "Morfoloji (Otsu)", rotation=90, va="center", ha="right", fontsize=11)
    fig.suptitle("Şekil 3. Segmentasyon yöntemleri (üst) ve morfolojik işlemler (alt) – D01, "
                 "ortak bağlam: log + Gauss5/keskinleştirme, min alan 400 px "
                 "(n: tüm kare, referans=23)", fontsize=10)
    fig.tight_layout()
    fig.savefig(REPORT_FIGS / "fig3_segmentation_morphology.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Parametre etkisi grafikleri (veri görüntüsü içermez)
# ---------------------------------------------------------------------------
def _stacked_errors(ax, d, labels):
    x = np.arange(len(d))
    ax.bar(x, d.missed, color="C3", label="kaçırılan")
    ax.bar(x, d.extra_from_split, bottom=d.missed, color="C1", label="bölünme (+)")
    ax.bar(x, d.false_pos, bottom=d.missed + d.extra_from_split, color="C4", label="yanlış pozitif")
    ax.bar(x, d.merged, bottom=d.missed + d.extra_from_split + d.false_pos, color="C7",
           label="birleşme")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Nesne hatası (10 karede toplam)")
    ax.set_yscale("symlog", linthresh=10)
    ax.legend(frameon=False, fontsize=7)


def _mae_bars(ax, d, labels, title):
    x = np.arange(len(d))
    ax.bar(x - .2, d.MAE, width=.4, label="MAE", color="C0")
    ax.bar(x + .2, d.Bias, width=.4, label="Bias", color="C2")
    ax.axhline(0, color="k", lw=.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yscale("symlog", linthresh=5)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=7)
    for i, v in enumerate(d.MAE):
        ax.text(i - .2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)


def param_stage(e, stage_prefix, fname, title):
    d = e[e.stage.str.startswith(stage_prefix)].reset_index(drop=True)
    labels = [f"{r.experiment_id}\n{r.changed}" for r in d.itertuples()]
    fig, ax = plt.subplots(1, 2, figsize=(max(10, 1.15 * len(d) * 2), 4.6))
    _mae_bars(ax[0], d, labels, f"{title}: geliştirme MAE ve Bias")
    _stacked_errors(ax[1], d, labels)
    ax[1].set_title("Hata bileşenleri")
    fig.tight_layout()
    fig.savefig(FIGURES / fname, dpi=140)
    plt.close(fig)


def param_threshold(e):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    c0 = e[e.stage == "7-Aday C0"].copy()
    c0["T"] = c0.changed.str.extract(r"T=(\d+)").astype(int)
    ax[0].plot(c0["T"], c0.MAE, "o-", label="MAE")
    ax[0].plot(c0["T"], c0.Bias, "s--", label="Bias")
    ax[0].plot(c0["T"], c0.missed / 10, "^:", label="kaçırılan / kare")
    ax[0].plot(c0["T"], c0.false_pos / 10, "v:", label="yanlış pozitif / kare")
    ax[0].axhline(0, color="k", lw=.6)
    ax[0].set_xlabel("Sabit eşik T (ham yoğunluk)")
    ax[0].set_title("Sabit eşik (Gauss 9x9, ham görüntü)")
    ax[0].legend(frameon=False, fontsize=7)
    ax[0].grid(alpha=.3)

    t3 = e[e.stage.str.contains("Tur1/3") & e.changed.str.startswith("Sabit")].copy()
    t3["T"] = t3.changed.str.extract(r"T=(\d+)").astype(int)
    ax[1].plot(t3["T"], t3.MAE, "o-")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Sabit eşik T (log dönüşümü sonrası ölçek)")
    ax[1].set_ylabel("MAE (log)")
    ax[1].set_title("Sabit eşik – log ölçeği (Tur 1)")
    ax[1].grid(alpha=.3)

    b = e[e.stage == "7-Aday B"].copy()
    b["block"] = b.changed.str.extract(r"blok=(\d+)").astype(int)
    b["C"] = b.changed.str.extract(r"C=(\d+)").astype(int)
    piv = b.pivot(index="block", columns="C", values="MAE")
    im = ax[2].imshow(piv.values, cmap="viridis_r")
    ax[2].set_xticks(range(len(piv.columns)))
    ax[2].set_xticklabels(piv.columns)
    ax[2].set_yticks(range(len(piv.index)))
    ax[2].set_yticklabels(piv.index)
    for (i, j), v in np.ndenumerate(piv.values):
        ax[2].text(j, i, f"{v:.1f}", ha="center", va="center", color="w", fontsize=9)
    ax[2].set_xlabel("C")
    ax[2].set_ylabel("Blok boyutu")
    ax[2].set_title("Adaptif eşik (Gauss 9x9) – geliştirme MAE")
    fig.colorbar(im, ax=ax[2], fraction=.046)
    fig.tight_layout()
    fig.savefig(FIGURES / "param_threshold.png", dpi=140)
    plt.close(fig)


def param_minarea_ws(e):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for rnd, mk in (("Tur1", "o-"), ("Tur2", "s--")):
        d = e[e.stage == f"{rnd}/6-MinAlan"].copy()
        d["a"] = d.changed.str.extract(r"(\d+)").astype(int)
        d = d.sort_values("a")
        ax[0].plot(d.a, d.MAE, mk, label=rnd)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Minimum nesne alanı (piksel)")
    ax[0].set_ylabel("Geliştirme MAE")
    ax[0].set_title("Minimum alan eşiği (seçilmeyen ayarlar)")
    ax[0].legend(frameon=False)
    ax[0].grid(alpha=.3)
    for rnd, mk in (("Tur1", "o-"), ("Tur2", "s--")):
        d = e[(e.stage == f"{rnd}/5-Ayirma") & e.changed.str.contains("min_dist")
              & ~e.changed.str.contains("sigma")].copy()
        d["md"] = d.changed.str.extract(r"min_dist=(\d+)").astype(int)
        d = d.sort_values("md")
        ax[1].plot(d.md, d.MAE, mk, label=rnd)
    ax[1].set_xlabel("Watershed min_distance (piksel)")
    ax[1].set_ylabel("Geliştirme MAE")
    ax[1].set_title("Distance transform + watershed")
    ax[1].legend(frameon=False)
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "param_minarea_ws.png", dpi=140)
    plt.close(fig)


def main():
    e = load_exp()
    param_stage(e, "Tur1/1-Yogunluk", "param_intensity.png", "Yoğunluk dönüşümü (Tur 1)")
    param_stage(e, "Tur1/2-Filtre", "param_filter.png", "Filtre (Tur 1)")
    param_stage(e, "Tur1/4-Morfoloji", "param_morphology.png", "Morfoloji (Tur 1)")
    param_threshold(e)
    param_minarea_ws(e)
    fig2()
    fig3()
    print("Şekiller üretildi.")


if __name__ == "__main__":
    main()
