"""
Bölüm 6-10 - Kontrollü deneyler (yalnızca GELİŞTİRME kümesi, 10 kare).

Tasarım: aşamalı tek-faktör (one-factor-at-a-time) karşılaştırma.
Her aşamada YALNIZCA bir bileşen değiştirilir; diğer bileşenler "bağlam"
(context) olarak sabit tutulur. Bir aşamanın en iyi ayarı (en düşük geliştirme
MAE'si; eşitlikte RMSE, sonra listedeki ilk sıra) bir sonraki aşamanın
bağlamına aktarılır.

  Aşama 1  Yoğunluk dönüşümü   (Bölüm 6)
  Aşama 2  Uzamsal filtre      (Bölüm 7)
  Aşama 3  Segmentasyon        (Bölüm 8)
  Aşama 4  Morfoloji ve sırası (Bölüm 9)
  Aşama 5  Temas eden hücrelerin ayrılması (distance transform + watershed)

Başlangıç bağlamı = Gauss 5x5 + Otsu + delik doldurma + 200 px minimum alan.
Bu en küçük "makul" zincir baseline hata analizinden türetilmiştir:
(i) benek parçalanması -> hafif yumuşatma, (ii) nükleol delikleri -> delik
doldurma, (iii) tek piksellik gürültü -> alan eşiği. Her birinin etkisi
ilgili aşamada ayrıca ölçülür.

Aşamalar iki tur çalıştırılır (koordinat iniş): 2. turda her bileşen,
diğer bileşenlerin 1. tur sonundaki en iyi ayarları sabitken yeniden sınanır.
Daha önce ölçülmüş bir konfigürasyon tekrar ölçülmez; tabloda "(tekrar)" olarak
gösterilir.

Not: Yumuşatma olmadan başlatılan ilk deneme (results/experiments_round0_raw_context.csv)
kırılgan bir yerel optimuma (CLAHE + Laplace keskinleştirme + sabit eşik) gitmiştir;
bu yüzden başlangıç bağlamı yukarıdaki gibi belirlenmiştir.

Çıktılar
  experiments.csv                     (repo kökü; Bölüm 18.2 zorunlu dosya)
  results/experiments_per_frame.csv   (her deney x her kare)
  results/stage_best.json             (her aşamada seçilen ayar)
"""

from __future__ import annotations

import copy
import json

import pandas as pd

from common import RESULTS, ROOT
from evaluate import evaluate
from pipeline import describe, make_config

FH = {"op": "fill_holes"}


def G(k):
    return {"method": "gaussian", "ksize": k}


def stage_definitions():
    """Her aşama: (aşama adı, değiştirilen anahtar, [(kısa ad, değer), ...])."""
    return [
        ("1-Yogunluk", "intensity", [
            ("orijinal", {"method": "none"}),
            ("contrast stretching", {"method": "stretch", "p_low": 0.0, "p_high": 99.5}),
            ("histogram esitleme", {"method": "histeq"}),
            ("CLAHE clip=2", {"method": "clahe", "clip": 2.0, "tiles": 8}),
            ("CLAHE clip=4", {"method": "clahe", "clip": 4.0, "tiles": 8}),
            ("histogram matching", {"method": "histmatch"}),
            ("log donusumu", {"method": "log"}),
        ]),
        ("2-Filtre", "filter", [
            ("filtre yok", {"method": "none"}),
            ("Gauss 3x3", G(3)),
            ("Gauss 5x5", G(5)),
            ("Gauss 9x9", G(9)),
            ("Median 5x5", {"method": "median", "ksize": 5}),
            ("Median 9x9", {"method": "median", "ksize": 9}),
            ("Laplace keskinlestirme", {"method": "sharpen", "amount": 1.0}),
            ("Gauss 5x5 + keskinlestirme", {"method": "gaussian+sharpen", "ksize": 5, "amount": 1.0}),
        ]),
        ("3-Segmentasyon", "segmentation", [
            ("Otsu", {"method": "otsu"}),
            # Eşikler, o anki yoğunluk dönüşümünün 0-255 ölçeğindedir.
            # Hem ham ölçeğe (10, 40) hem log ölçeğine (80-140) uygun değerler denenir.
            ("Sabit esik T=10", {"method": "fixed", "threshold": 10}),
            ("Sabit esik T=40", {"method": "fixed", "threshold": 40}),
            ("Sabit esik T=80", {"method": "fixed", "threshold": 80}),
            ("Sabit esik T=100", {"method": "fixed", "threshold": 100}),
            ("Sabit esik T=120", {"method": "fixed", "threshold": 120}),
            ("Sabit esik T=140", {"method": "fixed", "threshold": 140}),
            ("Adaptif (Gauss) blok=101 C=2", {"method": "adaptive", "block": 101, "C": 2}),
            ("Adaptif (Gauss) blok=101 C=10", {"method": "adaptive", "block": 101, "C": 10}),
            ("Adaptif (Gauss) blok=101 C=20", {"method": "adaptive", "block": 101, "C": 20}),
            ("Adaptif (Gauss) blok=51 C=10", {"method": "adaptive", "block": 51, "C": 10}),
            ("Adaptif (Gauss) blok=151 C=10", {"method": "adaptive", "block": 151, "C": 10}),
            ("Ucgen (triangle)", {"method": "triangle"}),
            ("K-means k=3 (2 on plan kumesi)", {"method": "kmeans", "k": 3, "bg_clusters": 1}),
        ]),
        ("4-Morfoloji", "morphology", [
            ("morfoloji yok", []),
            ("delik doldurma", [FH]),
            ("erosion r=3 + doldurma", [{"op": "erode", "size": 3}, FH]),
            ("dilation r=3 + doldurma", [{"op": "dilate", "size": 3}, FH]),
            ("opening r=3 + doldurma", [{"op": "open", "size": 3}, FH]),
            ("opening r=7 + doldurma", [{"op": "open", "size": 7}, FH]),
            ("closing r=3 + doldurma", [{"op": "close", "size": 3}, FH]),
            ("closing r=7 + doldurma", [{"op": "close", "size": 7}, FH]),
            ("opening r=3 -> closing r=3 + doldurma",
             [{"op": "open", "size": 3}, {"op": "close", "size": 3}, FH]),
            ("closing r=3 -> opening r=3 + doldurma",
             [{"op": "close", "size": 3}, {"op": "open", "size": 3}, FH]),
            ("doldurma -> opening r=7",
             [FH, {"op": "open", "size": 7}]),
        ]),
        ("5-Ayirma", "separation", [
            ("watershed yok", {"method": "none"}),
            ("watershed min_dist=10", {"method": "watershed", "min_distance": 10}),
            ("watershed min_dist=15", {"method": "watershed", "min_distance": 15}),
            ("watershed min_dist=20", {"method": "watershed", "min_distance": 20}),
            ("watershed min_dist=20, sigma=2",
             {"method": "watershed", "min_distance": 20, "smooth_sigma": 2}),
            ("watershed min_dist=25", {"method": "watershed", "min_distance": 25}),
        ]),
        ("6-MinAlan", "min_area", [
            ("min alan 50", 50),
            ("min alan 100", 100),
            ("min alan 200", 200),
            ("min alan 400", 400),
            ("min alan 800", 800),
        ]),
    ]


def main():
    context = make_config(filter=G(5), morphology=[FH], min_area=200)
    rows, per_frame, stage_best = [], [], {}
    seen = {}
    eid = 0

    plan = [(r, st) for r in (1, 2) for st in stage_definitions()]
    for rnd, (stage_name, key, variants) in plan:
        stage_name = f"Tur{rnd}/{stage_name}"
        stage_rows = []
        for short, value in variants:
            cfg = copy.deepcopy(context)
            cfg[key] = copy.deepcopy(value)
            sig = json.dumps(cfg, sort_keys=True)
            if sig in seen:  # aynı konfigürasyon önceki aşamada zaten ölçüldüyse tekrar etme
                prev = seen[sig]
                stage_rows.append({**prev, "stage": stage_name, "changed": short,
                                   "is_repeat_of": prev["experiment_id"]})
                continue
            eid += 1
            exp_id = f"E{eid:02d}"
            summ, per = evaluate(cfg, "development")
            row = {"experiment_id": exp_id, "stage": stage_name, "changed": short,
                   **describe(cfg), **{k: round(v, 4) if isinstance(v, float) else v
                                       for k, v in summ.items()},
                   "config_json": json.dumps(cfg, ensure_ascii=False), "is_repeat_of": ""}
            seen[sig] = row
            stage_rows.append(row)
            per["experiment_id"] = exp_id
            per_frame.append(per)
            print(f"{exp_id} [{stage_name}] {short:42s} MAE={summ['MAE']:.2f} "
                  f"RMSE={summ['RMSE']:.2f} Bias={summ['Bias']:+.2f} "
                  f"kacan={summ['missed']} bolunme+={summ['extra_from_split']} "
                  f"birlesme={summ['merged']} YP={summ['false_pos']}")
        st = pd.DataFrame(stage_rows)
        st["_order"] = range(len(st))
        best = st.sort_values(["MAE", "RMSE", "_order"]).iloc[0]
        stage_best[stage_name] = {"choice": best["changed"], "experiment_id": best["experiment_id"],
                                  "MAE": float(best["MAE"]), "RMSE": float(best["RMSE"])}
        context = json.loads(best["config_json"])
        print(f"  -> {stage_name} için seçilen: {best['changed']} "
              f"({best['experiment_id']}, MAE={best['MAE']:.2f})\n")
        rows.extend(r for r in stage_rows if not r.get("is_repeat_of"))
        # Tekrarlanan kontrol satırlarını da aşama tablosunda göstermek için kaydet
        for r in stage_rows:
            if r.get("is_repeat_of"):
                rows.append({**r, "experiment_id": f"{r['is_repeat_of']} (tekrar)"})

    # ---- Aşama 7: farklı segmentasyon ailelerinden aday pipeline'lar ----
    # C: koordinat inişin (Tur 1-2) vardığı nokta
    candidates = {"C": {"name": "Koordinat inis sonucu (sabit esik)",
                        "config": copy.deepcopy(context),
                        "experiment_id": stage_best[list(stage_best)[-1]]["experiment_id"]}}
    shared_post = {k: copy.deepcopy(context[k]) for k in ("morphology", "separation", "min_area")}
    LOG = {"method": "log"}
    CO_FH = [{"op": "close", "size": 3}, {"op": "open", "size": 3}, FH]
    families = {
        "A": ("Log + filtre + Otsu (otomatik global esik)",
              [(f"Otsu | {fn} | {mn} | {sn}",
                make_config(intensity=LOG, filter=fv, segmentation={"method": "otsu"},
                            morphology=mv, separation=sv, min_area=400))
               for fn, fv in (("Gauss 5x5", G(5)),
                              ("Gauss5+keskin", {"method": "gaussian+sharpen", "ksize": 5, "amount": 1.0}),
                              ("Median 9x9", {"method": "median", "ksize": 9}))
               for mn, mv in (("doldurma", [FH]), ("close3->open3+doldurma", CO_FH))
               for sn, sv in (("ws yok", {"method": "none"}),
                              ("ws md=15", {"method": "watershed", "min_distance": 15}))]),
        "B": ("Gauss + adaptif esik (yerel esik)",
              [(f"Adaptif blok={b} C={c}",
                make_config(filter=G(9), segmentation={"method": "adaptive", "block": b, "C": c},
                            **shared_post))
               for b in (51, 101, 151) for c in (1, 2, 3)]),
        "C0": ("Gauss + sabit esik (ham yogunluk) - karsilastirma",
              [(f"Sabit esik T={t}",
                make_config(filter=G(9), segmentation={"method": "fixed", "threshold": t},
                            **shared_post))
               for t in (4, 5, 6, 8, 10, 12)]),
    }
    for fam, (fam_name, variants) in families.items():
        fam_rows = []
        for short, cfg in variants:
            eid += 1
            exp_id = f"E{eid:02d}"
            summ, per = evaluate(cfg, "development")
            row = {"experiment_id": exp_id, "stage": f"7-Aday {fam}", "changed": short,
                   **describe(cfg), **{k: round(v, 4) if isinstance(v, float) else v
                                       for k, v in summ.items()},
                   "config_json": json.dumps(cfg, ensure_ascii=False), "is_repeat_of": ""}
            fam_rows.append(row)
            per["experiment_id"] = exp_id
            per_frame.append(per)
            print(f"{exp_id} [7-Aday {fam}] {short:42s} MAE={summ['MAE']:.2f} "
                  f"RMSE={summ['RMSE']:.2f} Bias={summ['Bias']:+.2f} "
                  f"kacan={summ['missed']} bolunme+={summ['extra_from_split']} "
                  f"birlesme={summ['merged']} YP={summ['false_pos']}")
        ft = pd.DataFrame(fam_rows)
        ft["_order"] = range(len(ft))
        best = ft.sort_values(["MAE", "RMSE", "_order"]).iloc[0]
        if fam == "C0":  # yalnızca karşılaştırma; aday C koordinat iniş sonucudur
            print(f"  -> Ham ölçekte en iyi sabit eşik: {best['changed']} (MAE={best['MAE']:.2f})\n")
            rows.extend(fam_rows)
            continue
        candidates[fam] = {"name": fam_name, "config": json.loads(best["config_json"]),
                           "experiment_id": best["experiment_id"]}
        print(f"  -> Aday {fam}: {best['changed']} ({best['experiment_id']}, MAE={best['MAE']:.2f})\n")
        rows.extend(fam_rows)
    candidates = {k: candidates[k] for k in ("A", "B", "C")}
    (RESULTS / "candidate_pipelines.json").write_text(
        json.dumps(candidates, indent=2, ensure_ascii=False))

    exp = pd.DataFrame(rows)
    cols = ["experiment_id", "stage", "changed", "Yogunluk", "Filtre", "Segmentasyon",
            "Morfoloji", "MinAlan", "MAE", "RMSE", "Bias", "ExactAcc", "missed",
            "extra_from_split", "merged", "false_pos", "mean_time_s", "config_json"]
    exp[cols].to_csv(ROOT / "experiments.csv", index=False)
    pd.concat(per_frame).to_csv(RESULTS / "experiments_per_frame.csv", index=False)
    (RESULTS / "stage_best.json").write_text(json.dumps(stage_best, indent=2, ensure_ascii=False))
    n_unique = exp[~exp.experiment_id.str.contains("tekrar")].shape[0]
    print(f"Toplam benzersiz deney: {n_unique}")


if __name__ == "__main__":
    main()
