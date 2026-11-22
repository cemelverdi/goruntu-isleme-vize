"""Bir pipeline konfigürasyonunu geliştirme veya doğrulama karelerinde değerlendirir."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import tifffile

from common import RESULTS, read_image, tra_path
from metrics import count_metrics, error_decomposition
from pipeline import run_pipeline

# Histogram matching için referans görüntü: yalnızca geliştirme kümesinden (Bölüm 6).
# İlk geliştirme karesi (D01, t001) seçim sırasına göre sabit olarak kullanılır.
HISTMATCH_REF = ("01", 1)


@lru_cache(maxsize=None)
def _img(seq: str, t: int) -> np.ndarray:
    return read_image(seq, t)


@lru_cache(maxsize=None)
def _markers(seq: str, t: int) -> np.ndarray:
    assert seq == "01", "Deneylerde yalnızca sekans 01 referansları kullanılır"
    return tifffile.imread(tra_path(seq, t))


def hm_ref() -> np.ndarray:
    return _img(*HISTMATCH_REF)


def references(group: str) -> pd.DataFrame:
    ref = pd.read_csv(RESULTS / "reference_counts_devval.csv", dtype={"sequence": str})
    return ref[ref.group == group].reset_index(drop=True)


def evaluate(cfg: dict, group: str = "development", decompose: bool = True) -> tuple[dict, pd.DataFrame]:
    ref = references(group)
    rows = []
    for _, r in ref.iterrows():
        t = int(r.frame_index)
        out = run_pipeline(_img(r.sequence, t), cfg, ref_img=hm_ref())
        row = {"sample_id": r.sample_id, "frame_index": t,
               "reference_count": int(r.reference_count), "pred_count": out["count"],
               "error": out["count"] - int(r.reference_count), "time_s": out["time_s"]}
        if decompose:
            row.update(error_decomposition(out["labels"], _markers(r.sequence, t)))
        rows.append(row)
    per = pd.DataFrame(rows)
    summ = {**count_metrics(per.reference_count, per.pred_count),
            "mean_time_s": float(per.time_s.mean())}
    if decompose:
        for c in ("missed", "extra_from_split", "merged", "false_pos"):
            summ[c] = int(per[c].sum())
    return summ, per
