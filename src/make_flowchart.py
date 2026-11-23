"""Şekil 1 - Nihai pipeline akış diyagramı (veri görüntüsü içermez)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import FIGURES

STEPS = [
    ("Girdi", "uint8 floresan görüntü\n1024×1024", "#E8E8E8"),
    ("1. Yoğunluk", "Log dönüşümü\nlog(1+I)/log(256)", "#D6E4F0"),
    ("2. Filtre", "Gauss 5×5\n+ Laplace keskinleştirme\n(3×3, α=1)", "#D6E4F0"),
    ("3. Segmentasyon", "Otsu eşikleme\n(görüntü başına\notomatik eşik)", "#DCEFD9"),
    ("4. Morfoloji", "Closing r=3\n→ Opening r=3\n→ Delik doldurma", "#FBE6C8"),
    ("5. Ayırma", "Distance transform\n+ watershed\n(min_distance=15)", "#FBE6C8"),
    ("6. Filtre/Sayım", "Alan < 400 px sil\n→ bileşen sayısı", "#F3D9D9"),
]


def main():
    fig, ax = plt.subplots(figsize=(13, 2.7))
    ax.set_xlim(0, len(STEPS) * 1.9)
    ax.set_ylim(0, 2.4)
    ax.axis("off")
    w, h = 1.55, 1.75
    for i, (title, body, color) in enumerate(STEPS):
        x = 0.15 + i * 1.9
        ax.add_patch(FancyBboxPatch((x, 0.3), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=color, ec="#555555", lw=1))
        ax.text(x + w / 2, 0.3 + h - 0.22, title, ha="center", va="center",
                fontsize=10, fontweight="bold")
        ax.text(x + w / 2, 0.3 + (h - 0.4) / 2, body, ha="center", va="center", fontsize=8.6)
        if i < len(STEPS) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.02, 0.3 + h / 2), (x + 1.9 - 0.02, 0.3 + h / 2),
                                         arrowstyle="-|>", mutation_scale=14, color="#333333"))
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_flowchart.png", dpi=200)
    plt.close(fig)
    print("Şekil 1 kaydedildi.")


if __name__ == "__main__":
    main()
