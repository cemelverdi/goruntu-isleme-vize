"""
Bölüm 12 - Nihai pipeline'ın sabitlenmesi ve test tahminlerinin kilitlenmesi.

final_pipeline.json'daki nihai yöntem ve baseline, 15 test karesinde
(Sequence 02) çalıştırılır; tahminler test_predictions_locked.csv dosyasına
yazılır. Bu betik test referanslarına (02_GT) HİÇ erişmez.

Dosya oluştuktan sonra betik yeniden çalışmayı reddeder. Kilitleme ayrı bir
GitHub commit'i ile kaydedilir; test metrikleri ondan sonraki bir commit'tedir.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from common import ROOT, load_selection, read_image
from evaluate import hm_ref
from pipeline import run_pipeline

OUT = ROOT / "test_predictions_locked.csv"
FINAL = ROOT / "final_pipeline.json"
REPEATS = 5


def timed(img, cfg):
    times, out = [], None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        out = run_pipeline(img, cfg, ref_img=hm_ref())
        times.append(time.perf_counter() - t0)
    return out["count"], float(np.median(times))


def main():
    if OUT.exists():
        raise SystemExit("test_predictions_locked.csv zaten var: test tahminleri kilitli, değiştirilemez.")
    final_bytes = FINAL.read_bytes()
    final = json.loads(final_bytes)
    assert final.get("locked"), "final_pipeline.json sabitlenmemiş"
    sha = hashlib.sha256(final_bytes).hexdigest()

    test = load_selection(["test"])
    assert set(test.sequence) == {"02"} and len(test) == 15
    rows = []
    for _, r in test.iterrows():
        img = read_image("02", int(r.frame_index))
        fc, ft = timed(img, final["pipeline"])
        bc, bt = timed(img, final["baseline"]["pipeline"])
        rows.append({"sample_id": r.sample_id, "test_block": r.test_block,
                     "sequence": "02", "frame_index": int(r.frame_index),
                     "image_file": r.image_file,
                     "final_pred_count": fc, "baseline_pred_count": bc,
                     "final_time_s": round(ft, 4), "baseline_time_s": round(bt, 4)})
    df = pd.DataFrame(rows)
    df["final_pipeline_sha256"] = sha
    df["locked_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    df.to_csv(OUT, index=False)
    print(df.drop(columns=["final_pipeline_sha256", "image_file"]).to_string(index=False))
    print(f"\nKilitlendi -> {OUT.name}  (final_pipeline.json sha256: {sha[:16]}...)")


if __name__ == "__main__":
    main()
