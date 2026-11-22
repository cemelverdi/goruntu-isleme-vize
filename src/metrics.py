"""Sayım performans metrikleri (Bölüm 11 ve 13)."""

from __future__ import annotations

import numpy as np


def error_decomposition(pred_labels: np.ndarray, ref_markers: np.ndarray) -> dict:
    """Sayım hatasını nesne düzeyinde ayrıştırır (yalnızca geliştirme/doğrulama için).

    missed : hiçbir tahmin bileşeniyle örtüşmeyen referans hücre
    split  : birden fazla tahmin bileşeniyle örtüşen referans hücre
             (extra_from_split = fazladan sayılan parça sayısı)
    merged : birden fazla referans hücreyi kapsayan tahmin bileşeni
    false_pos : hiçbir referans hücreyle örtüşmeyen tahmin bileşeni
    """
    fg = (pred_labels > 0) & (ref_markers > 0)
    pairs = np.unique(np.stack([ref_markers[fg], pred_labels[fg]]), axis=1)
    ref_ids = np.unique(ref_markers[ref_markers > 0])
    pred_ids = np.unique(pred_labels[pred_labels > 0])
    per_ref = {r: 0 for r in ref_ids}
    per_pred = {p: 0 for p in pred_ids}
    for r, p in pairs.T:
        per_ref[r] += 1
        per_pred[p] += 1
    return {
        "missed": sum(v == 0 for v in per_ref.values()),
        "split": sum(v > 1 for v in per_ref.values()),
        "extra_from_split": sum(v - 1 for v in per_ref.values() if v > 1),
        "merged": sum(v > 1 for v in per_pred.values()),
        "false_pos": sum(v == 0 for v in per_pred.values()),
    }


def count_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE, Bias (tahmin - gerçek ortalaması) ve tam doğru sayım oranı."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "Bias": float(np.mean(err)),
        "ExactAcc": float(np.mean(err == 0)),
    }
