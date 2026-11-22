"""Ortak yardımcılar: yollar, görüntü okuma, seçim dosyası."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "Fluo-N2DH-GOWT1"
RESULTS = ROOT / "results"          # sayısal sonuçlar (repoya girer)
FIGURES = ROOT / "figures"          # veri görüntüsü içermeyen grafikler (repoya girer)
REPORT_FIGS = ROOT / "report_figures"  # veri görüntüsü içeren şekiller (yalnızca rapor, repoya girmez)
SELECTED = ROOT / "selected_frames.csv"

for _d in (RESULTS, FIGURES, REPORT_FIGS):
    _d.mkdir(exist_ok=True)


def image_path(seq: str, t: int, data_root: Path = DATA_ROOT) -> Path:
    return data_root / seq / f"t{t:03d}.tif"


def tra_path(seq: str, t: int, data_root: Path = DATA_ROOT) -> Path:
    return data_root / f"{seq}_GT" / "TRA" / f"man_track{t:03d}.tif"


def read_image(seq: str, t: int, data_root: Path = DATA_ROOT) -> np.ndarray:
    return tifffile.imread(image_path(seq, t, data_root))


def load_selection(groups: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(SELECTED, dtype={"sequence": str})
    if groups is not None:
        df = df[df["group"].isin(groups)].reset_index(drop=True)
    return df


def display_stretch(img: np.ndarray, p_low=0.0, p_high=99.8) -> np.ndarray:
    """Yalnızca GÖRSELLEŞTİRME için yüzdelik germe (işleme zincirinde kullanılmaz)."""
    lo, hi = np.percentile(img, [p_low, p_high])
    return np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
