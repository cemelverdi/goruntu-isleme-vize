"""
Bölüm 14 - Zamansal analiz (+ Şekil 5 ve Şekil 6).

Test kilitlendikten SONRA çalıştırılır. Her test bloğu için kare numarasına
göre hücre sayısı (referans, baseline, nihai) çizilir. Referanstaki ani
değişimler, Sequence 02 iz dosyası (02_GT/TRA/man_track.txt: L B E P) ile
açıklanır:
ardışık karelerdeki etiket kümeleri karşılaştırılır:
  - bolunme_yavrusu: ebeveyni (P>0) olan ve izi bu karede başlayan hücre
  - bolunen_ana    : bu karede kaybolan ve yavruları olan hücre (mitoz)
  - yeni_giren     : izi bu karede başlayan, ebeveynsiz hücre (görüntüye giriş)
  - yeniden_beliren: izi daha önce başlamış ama önceki karede işaretlenmemiş hücre
  - kaybolan       : bu karede işaretlenmemiş hücre (çıkış / geçici kayıp)

Çıktılar
  figures/fig5_temporal.png            Şekil 5 (yalnızca grafik, repoya girer)
  results/temporal_events.csv          blok içi olaylar ve sayım değişimleri
  report_figures/fig6_examples.png     Şekil 6: başarılı / hatalı / zor örnekler (yalnızca rapor)
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from skimage.segmentation import find_boundaries

from common import DATA_ROOT, FIGURES, REPORT_FIGS, RESULTS, ROOT, display_stretch, read_image, tra_path
from evaluate import hm_ref
from pipeline import run_pipeline


def load_tracks() -> pd.DataFrame:
    return pd.read_csv(DATA_ROOT / "02_GT" / "TRA" / "man_track.txt", sep=r"\s+",
                       header=None, names=["L", "B", "E", "P"])


def _labels_in(t: int) -> set[int]:
    m = tifffile.imread(tra_path("02", t))
    return set(np.unique(m[m > 0]).tolist())


def events(df: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    """Ardışık iki karedeki referans etiket kümelerini karşılaştırarak olayları sınıflandırır."""
    parent = dict(zip(tracks.L, tracks.P))
    rows = []
    for b, d in df.groupby("test_block", sort=True):
        d = d.sort_values("frame_index")
        prev = None
        for _, r in d.iterrows():
            t = int(r.frame_index)
            cur = _labels_in(t)
            row = {"sample_id": r.sample_id, "test_block": b, "frame_index": t,
                   "reference_count": int(r.reference_count),
                   "delta_ref": np.nan, "delta_final": np.nan, "delta_baseline": np.nan,
                   "bolunme_yavrusu": 0, "yeni_giren": 0, "yeniden_beliren": 0,
                   "bolunen_ana": 0, "kaybolan": 0}
            if prev is not None:
                lab_prev, pr = prev
                appeared, gone = cur - lab_prev, lab_prev - cur
                parents = set(tracks.P[tracks.P > 0])
                for l in appeared:
                    first = int(tracks.loc[tracks.L == l, "B"].iloc[0])
                    if parent.get(l, 0) > 0 and first == t:
                        row["bolunme_yavrusu"] += 1
                    else:
                        row["yeni_giren" if first == t else "yeniden_beliren"] += 1
                for l in gone:
                    row["bolunen_ana" if l in parents else "kaybolan"] += 1
                row["delta_ref"] = r.reference_count - pr.reference_count
                row["delta_final"] = r.final_pred_count - pr.final_pred_count
                row["delta_baseline"] = r.baseline_pred_count - pr.baseline_pred_count
            rows.append(row)
            prev = (cur, r)
    ev = pd.DataFrame(rows)
    # Tutarlılık: delta_ref = yavru + giren + yeniden beliren - bölünen ana - kaybolan
    chk = ev.dropna(subset=["delta_ref"])
    assert (chk.delta_ref == chk.bolunme_yavrusu + chk.yeni_giren + chk.yeniden_beliren
            - chk.bolunen_ana - chk.kaybolan).all()
    return ev


def fig5(df: pd.DataFrame, ev: pd.DataFrame):
    blocks = sorted(df.test_block.unique())
    fig, ax = plt.subplots(1, len(blocks), figsize=(5.2 * len(blocks), 4.3), sharey=True)
    for a, b in zip(ax, blocks):
        d = df[df.test_block == b]
        a.plot(d.frame_index, d.reference_count, "ko-", lw=2, label="Referans (TRA)")
        a.plot(d.frame_index, d.final_pred_count, "s-", color="C0", label="Nihai pipeline")
        a.plot(d.frame_index, d.baseline_pred_count, "^--", color="C3", label="Baseline")
        e = ev[ev.test_block == b]
        for _, r in e.iterrows():
            notes = []
            if r.bolunen_ana:
                notes.append(f"ana hücre bölünmeye girdi (-{r.bolunen_ana})")
            if r.bolunme_yavrusu:
                notes.append(f"+{r.bolunme_yavrusu} bölünme yavrusu")
            if r.yeni_giren:
                notes.append(f"+{r.yeni_giren} giriş")
            if r.yeniden_beliren:
                notes.append(f"+{r.yeniden_beliren} yeniden beliren")
            if r.kaybolan:
                notes.append(f"-{r.kaybolan} kaybolan")
            if notes:
                a.annotate("\n".join(notes), (r.frame_index, r.reference_count),
                           textcoords="offset points", xytext=(0, 10), ha="center",
                           fontsize=7.5, color="k")
        a.set_title(f"Blok {b} (kareler {d.frame_index.min()}–{d.frame_index.max()})")
        a.set_xlabel("Kare numarası")
        a.set_xticks(d.frame_index)
        a.grid(alpha=.3)
    ax[0].set_ylabel("Hücre sayısı")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Şekil 5. Test bloklarında hücre sayısının zamana göre değişimi "
                 "(olaylar: Sequence 02 iz dosyasından)", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIGURES / "fig5_temporal.png", dpi=150)
    plt.close(fig)


def overlay(img, labels, markers):
    """Gri görüntü + tahmin sınırları (yeşil); hiçbir tahminle örtüşmeyen referanslar kırmızı."""
    rgb = np.dstack([display_stretch(img, 0, 99.5)] * 3)
    rgb[find_boundaries(labels, mode="outer")] = (0.1, 1, 0.1)
    missed = np.zeros_like(markers, bool)
    for lab in np.unique(markers[markers > 0]):
        m = markers == lab
        if not (labels[m] > 0).any():
            missed |= m
    rgb[missed] = rgb[missed] * 0.3 + np.array([1, 0, 0]) * 0.7
    return rgb, int(len(np.unique(markers[missed])) if missed.any() else 0)


def fig6(df: pd.DataFrame, final: dict):
    final_err = df.final_error.abs()
    best = df.loc[final_err.idxmin()]
    worst = df.loc[final_err.idxmax()]
    base_worst = df.loc[df.baseline_error.abs().idxmax()]
    panels = [
        (best, final["pipeline"], "final_pred_count", "Başarılı örnek – nihai pipeline"),
        (worst, final["pipeline"], "final_pred_count", "En hatalı örnek – nihai pipeline"),
        (base_worst, final["baseline"]["pipeline"], "baseline_pred_count",
         "Zor örnek – baseline (aynı kare: nihai yöntem sağda)"),
        (base_worst, final["pipeline"], "final_pred_count", "Aynı kare – nihai pipeline"),
    ]
    fig, ax = plt.subplots(1, 4, figsize=(22, 6))
    for a, (r, cfg, col, title) in zip(ax, panels):
        t = int(r.frame_index)
        img = read_image("02", t)
        out = run_pipeline(img, cfg, ref_img=hm_ref())
        assert out["count"] == r[col]
        rgb, n_miss = overlay(img, out["labels"], tifffile.imread(tra_path("02", t)))
        a.imshow(rgb)
        a.set_title(f"{title}\n{r.sample_id} (t{t:03d}): tahmin={out['count']}, "
                    f"referans={r.reference_count}, kaçırılan={n_miss}", fontsize=9.5)
        a.axis("off")
    fig.suptitle("Şekil 6. Test karelerinden örnekler – yeşil: tahmin sınırları, "
                 "kırmızı: hiçbir tahminle örtüşmeyen referans hücreler", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(REPORT_FIGS / "fig6_examples.png", dpi=100)
    plt.close(fig)


def main():
    df = pd.read_csv(ROOT / "test_results.csv")
    final = json.loads((ROOT / "final_pipeline.json").read_text())
    ev = events(df, load_tracks())
    ev.to_csv(RESULTS / "temporal_events.csv", index=False)
    print(ev.to_string(index=False))
    fig5(df, ev)
    fig6(df, final)
    print("Şekil 5 ve 6 kaydedildi.")


if __name__ == "__main__":
    main()
