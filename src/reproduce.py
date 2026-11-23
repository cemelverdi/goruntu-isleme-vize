"""
Tekrarlanabilirlik yardımcıları (main.ipynb tarafından kullanılır).

Kilitli aşamalar (veri seçimi, baseline min alanı, nihai pipeline, test
tahminleri) yeniden ÜRETİLMEZ; aynı kod yeniden çalıştırılarak kayıtlı
dosyalarla birebir aynı sonucun elde edildiği DOĞRULANIR.
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from common import DATA_ROOT, RESULTS, ROOT, load_selection, read_image
from evaluate import hm_ref, references
from pipeline import run_pipeline


def run_main(module, *args: str):
    """Bir betiğin main() fonksiyonunu komut satırı argümanlarıyla çağırır."""
    old = sys.argv
    sys.argv = [module.__name__, *args]
    try:
        module.main()
    finally:
        sys.argv = old


def check_selection() -> pd.DataFrame:
    import select_frames
    new = select_frames.select_frames(DATA_ROOT)
    saved = pd.read_csv(ROOT / "selected_frames.csv", dtype={"sequence": str})
    cols = ["sample_id", "group", "sequence", "frame_index", "test_block", "seed"]
    a = new[cols].fillna("").astype(str).reset_index(drop=True)
    b = saved[cols].fillna("").astype(str).reset_index(drop=True)
    assert a.equals(b), "Seçim kayıtlı selected_frames.csv ile uyuşmuyor!"
    print(f"✓ Veri seçimi aynı (seed={select_frames.SEED}).")
    return saved


def check_validation() -> pd.DataFrame:
    final = json.loads((ROOT / "final_pipeline.json").read_text())
    cands = {f"Aday {k}": v for k, v in final["all_candidates"].items()}
    cands["Baseline"] = final["baseline"]["pipeline"]
    saved = pd.read_csv(ROOT / "validation_results.csv")
    ref = references("validation")
    for name, cfg in cands.items():
        preds = [run_pipeline(read_image("01", int(t)), cfg, ref_img=hm_ref())["count"]
                 for t in ref.frame_index]
        rec = saved.loc[saved.Pipeline == name, "Tahminler"].iloc[0]
        assert " ".join(map(str, preds)) == rec, f"{name}: doğrulama tahminleri farklı"
    sel = saved.loc[saved.Secildi, "Pipeline"].iloc[0]
    assert sel == f"Aday {final['selected_candidate']}"
    print(f"✓ Doğrulama tahminleri aynı; seçilen: {sel}.")
    return saved


def check_locked_test() -> pd.DataFrame:
    final = json.loads((ROOT / "final_pipeline.json").read_text())
    locked = pd.read_csv(ROOT / "test_predictions_locked.csv")
    for _, r in locked.iterrows():
        img = read_image("02", int(r.frame_index))
        assert run_pipeline(img, final["pipeline"], ref_img=hm_ref())["count"] == r.final_pred_count
        assert (run_pipeline(img, final["baseline"]["pipeline"], ref_img=hm_ref())["count"]
                == r.baseline_pred_count)
    print("✓ 15 test karesinde nihai ve baseline tahminleri kilitli değerlerle aynı.")
    return locked


def check_experiments(tol: float = 1e-9) -> pd.DataFrame:
    """experiments.csv'yi yeniden üretir ve sayım metriklerini önceki sürümle karşılaştırır."""
    import run_experiments
    old = pd.read_csv(ROOT / "experiments.csv")
    run_experiments.main()
    new = pd.read_csv(ROOT / "experiments.csv")
    cols = ["experiment_id", "MAE", "RMSE", "Bias", "missed", "extra_from_split", "merged", "false_pos"]
    assert (old[cols].round(6).equals(new[cols].round(6))), "Deney metrikleri değişti!"
    print(f"✓ {len(new)} deney satırı yeniden üretildi; metrikler aynı.")
    return new


__all__ = ["run_main", "check_selection", "check_validation", "check_locked_test",
           "check_experiments", "RESULTS", "ROOT", "load_selection"]
