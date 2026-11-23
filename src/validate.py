"""
Bölüm 11 - Doğrulama ve nihai yöntem seçimi.

Üç aday pipeline (results/candidate_pipelines.json) 5 doğrulama karesinde
çalıştırılır. Seçim kuralı (ödev metni):
  1. En düşük doğrulama MAE'si
  2. Eşitlikte: RMSE  ->  ortalama işlem süresi  ->  yöntem karmaşıklığı (adım sayısı)

Süre ölçümü: her kare REPEATS kez çalıştırılır, kare başına medyan süre alınır.

Çıktılar
  validation_results.csv              (repo kökü; Bölüm 18.2 zorunlu)
  results/validation_per_frame.csv
  final_pipeline.json                 (repo kökü; nihai pipeline + baseline, SABİTLENMİŞ)
Baseline da referans olarak doğrulama karelerinde raporlanır (aday değildir).
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import tifffile

from common import RESULTS, ROOT, read_image, tra_path
from evaluate import HISTMATCH_REF, hm_ref, references
from metrics import count_metrics, error_decomposition
from pipeline import make_config, run_pipeline
from select_frames import SEED

REPEATS = 5
FINAL_PATH = ROOT / "final_pipeline.json"


def complexity(cfg: dict) -> int:
    """Yöntem karmaşıklığı: etkin işlem adımı sayısı."""
    n = 1  # segmentasyon
    n += cfg["intensity"]["method"] != "none"
    n += cfg["filter"]["method"] != "none"
    n += len(cfg["morphology"])
    n += cfg["separation"]["method"] != "none"
    n += cfg["min_area"] > 0
    return int(n)


def timed_run(img, cfg):
    times, out = [], None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        out = run_pipeline(img, cfg, ref_img=hm_ref())
        times.append(time.perf_counter() - t0)
    return out, float(np.median(times))


def main():
    if FINAL_PATH.exists():
        raise SystemExit(f"{FINAL_PATH.name} zaten var: nihai pipeline sabitlenmiş, yeniden seçim yapılmaz.")

    cands = json.loads((RESULTS / "candidate_pipelines.json").read_text())
    baseline_cfg = json.loads((RESULTS / "baseline_config.json").read_text())
    methods = {f"Aday {k}": v["config"] for k, v in cands.items()}
    methods["Baseline"] = make_config(min_area=baseline_cfg["min_area"])  # pipeline ile birebir aynı

    ref = references("validation")
    rows = []
    for name, cfg in methods.items():
        for _, r in ref.iterrows():
            t = int(r.frame_index)
            out, sec = timed_run(read_image("01", t), cfg)
            rows.append({"method": name, "sample_id": r.sample_id, "frame_index": t,
                         "reference_count": int(r.reference_count), "pred_count": out["count"],
                         "error": out["count"] - int(r.reference_count), "time_s": sec,
                         **error_decomposition(out["labels"], tifffile.imread(tra_path("01", t)))})
    per = pd.DataFrame(rows)
    per.to_csv(RESULTS / "validation_per_frame.csv", index=False)

    summ = []
    for name, d in per.groupby("method", sort=False):
        m = count_metrics(d.reference_count, d.pred_count)
        summ.append({"Pipeline": name,
                     "Aciklama": cands[name[-1]]["name"] if name.startswith("Aday") else "Otsu + CCL",
                     "Val_MAE": round(m["MAE"], 3), "Val_RMSE": round(m["RMSE"], 3),
                     "Val_Bias": round(m["Bias"], 3), "Val_ExactAcc": round(m["ExactAcc"], 3),
                     "Ort_Sure_s": round(d.time_s.mean(), 4),
                     "Adim_Sayisi": complexity(methods[name]),
                     "kacan": int(d.missed.sum()), "bolunme_fazla": int(d.extra_from_split.sum()),
                     "birlesme": int(d.merged.sum()), "yanlis_pozitif": int(d.false_pos.sum()),
                     "Tahminler": " ".join(map(str, d.pred_count))})
    tab = pd.DataFrame(summ)
    cand_tab = tab[tab.Pipeline.str.startswith("Aday")].sort_values(
        ["Val_MAE", "Val_RMSE", "Ort_Sure_s", "Adim_Sayisi"]).reset_index(drop=True)
    winner = cand_tab.iloc[0]
    tab["Secildi"] = tab.Pipeline == winner.Pipeline
    tab.to_csv(ROOT / "validation_results.csv", index=False)

    print("Referans (doğrulama):", ref.reference_count.tolist())
    print(tab.drop(columns=["Aciklama"]).to_string(index=False))
    print(f"\nSeçilen: {winner.Pipeline} – {winner.Aciklama}")

    key = winner.Pipeline[-1]
    final = {
        "locked": True,
        "locked_note": "Bu dosya doğrulama sonrası sabitlenmiştir; test sonuçlarından sonra değiştirilemez.",
        "seed": SEED,
        "selection_rule": "min validation MAE; ties -> RMSE -> mean time -> number of steps",
        "selected_candidate": key,
        "selected_name": cands[key]["name"],
        "development_experiment_id": cands[key]["experiment_id"],
        "validation_metrics": {k: float(winner[k]) for k in ("Val_MAE", "Val_RMSE", "Val_Bias",
                                                               "Val_ExactAcc", "Ort_Sure_s")},
        "pipeline": methods[winner.Pipeline],
        "histmatch_reference_frame": list(HISTMATCH_REF),
        "baseline": {"pipeline": methods["Baseline"], **baseline_cfg},
        "all_candidates": {k: v["config"] for k, v in cands.items()},
    }
    FINAL_PATH.write_text(json.dumps(final, indent=2, ensure_ascii=False))
    print(f"Sabitlendi -> {FINAL_PATH.name}")


if __name__ == "__main__":
    main()
