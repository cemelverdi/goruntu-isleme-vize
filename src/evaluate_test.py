"""
Bölüm 13 - Test sonuçlarının değerlendirilmesi.

Kilitli tahminler (test_predictions_locked.csv) test referanslarıyla
karşılaştırılır. Tahminler YENİDEN ÜRETİLMEZ ve DEĞİŞTİRİLMEZ; yalnızca
nesne düzeyindeki hata ayrıştırması için nihai pipeline tekrar çalıştırılır ve
sayının kilitli tahminle birebir aynı olduğu doğrulanır.

Çıktılar
  test_results.csv                   (repo kökü; kare bazında, Bölüm 18.2 zorunlu)
  results/test_summary.csv           (yöntem bazında özet metrikler)
  results/test_block_summary.csv     (blok bazında özet)
"""

from __future__ import annotations

import json

import pandas as pd
import tifffile

from common import RESULTS, ROOT, read_image, tra_path
from evaluate import hm_ref
from metrics import count_metrics, error_decomposition
from pipeline import run_pipeline


def main():
    pred = pd.read_csv(ROOT / "test_predictions_locked.csv")
    ref = pd.read_csv(RESULTS / "reference_counts_test.csv")
    final = json.loads((ROOT / "final_pipeline.json").read_text())

    df = pred.merge(ref[["sample_id", "reference_count"]], on="sample_id", validate="1:1")
    df["baseline_error"] = df.baseline_pred_count - df.reference_count
    df["final_error"] = df.final_pred_count - df.reference_count

    # Nesne düzeyinde tanı (tahminler değişmez; sadece doğrulanır)
    diag = []
    for _, r in df.iterrows():
        img = read_image("02", int(r.frame_index))
        mk = tifffile.imread(tra_path("02", int(r.frame_index)))
        d = {}
        for name, cfg, col in (("final", final["pipeline"], "final_pred_count"),
                               ("baseline", final["baseline"]["pipeline"], "baseline_pred_count")):
            out = run_pipeline(img, cfg, ref_img=hm_ref())
            assert out["count"] == r[col], f"{r.sample_id}: kilitli tahminle uyuşmazlık ({name})"
            d.update({f"{name}_{k}": v for k, v in error_decomposition(out["labels"], mk).items()})
        diag.append(d)
    df = pd.concat([df, pd.DataFrame(diag)], axis=1)

    cols = ["sample_id", "test_block", "frame_index", "reference_count",
            "baseline_pred_count", "final_pred_count", "baseline_error", "final_error",
            "baseline_time_s", "final_time_s",
            "final_missed", "final_extra_from_split", "final_merged", "final_false_pos",
            "baseline_missed", "baseline_extra_from_split", "baseline_merged", "baseline_false_pos"]
    df[cols].to_csv(ROOT / "test_results.csv", index=False)

    summ = []
    for name in ("baseline", "final"):
        m = count_metrics(df.reference_count, df[f"{name}_pred_count"])
        summ.append({"Yontem": "Baseline" if name == "baseline" else "Nihai pipeline",
                     **{k: round(v, 3) for k, v in m.items()},
                     "Ort_Sure_s": round(df[f"{name}_time_s"].mean(), 4),
                     "kacan": int(df[f"{name}_missed"].sum()),
                     "bolunme_fazla": int(df[f"{name}_extra_from_split"].sum()),
                     "birlesme": int(df[f"{name}_merged"].sum()),
                     "yanlis_pozitif": int(df[f"{name}_false_pos"].sum())})
    summ = pd.DataFrame(summ)
    summ.to_csv(RESULTS / "test_summary.csv", index=False)

    blk = []
    for b, d in df.groupby("test_block"):
        for name in ("baseline", "final"):
            m = count_metrics(d.reference_count, d[f"{name}_pred_count"])
            blk.append({"Blok": b, "Kareler": f"{d.frame_index.min()}-{d.frame_index.max()}",
                        "Yontem": name, **{k: round(v, 3) for k, v in m.items()}})
    pd.DataFrame(blk).to_csv(RESULTS / "test_block_summary.csv", index=False)

    print(df[cols[:8] + ["final_missed", "final_extra_from_split", "final_merged",
                         "final_false_pos"]].to_string(index=False))
    print("\n", summ.to_string(index=False))
    print("\n", pd.DataFrame(blk).to_string(index=False))


if __name__ == "__main__":
    main()
