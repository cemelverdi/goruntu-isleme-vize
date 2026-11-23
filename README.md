# Klasik Görüntü İşleme ile Mikroskobik Hücre Segmentasyonu, Sayımı ve Zamansal Analizi

Görüntü İşleme (Yüksek Lisans) ara sınav projesi.

Floresan mikroskopi görüntülerindeki hücre çekirdekleri **yalnızca klasik görüntü işleme yöntemleriyle** bölütlenir ve sayılır; sayının zaman içindeki değişimi incelenir. Derin öğrenme ya da önceden eğitilmiş model kullanılmamıştır.

## Sonuç özeti

**Nihai pipeline** (`final_pipeline.json`):

```
Log dönüşümü → Gauss 5×5 + Laplace keskinleştirme → Otsu eşikleme
→ closing (r=3) → opening (r=3) → delik doldurma
→ distance transform + watershed (min_distance=15) → min alan 400 px → sayım
```

| Küme | Yöntem | MAE | RMSE | Bias | Süre (s/kare) |
|---|---|---|---|---|---|
| Doğrulama (5 kare) | Baseline | 3.40 | 3.92 | +3.40 | 0.02 |
| Doğrulama (5 kare) | **Nihai** | **1.20** | **1.26** | +1.20 | 0.22 |
| Test (15 kare) | Baseline | 8.20 | 8.99 | −1.00 | 0.02 |
| Test (15 kare) | **Nihai** | **2.67** | **2.78** | −2.67 | 0.19 |

Test kümesinde nihai yöntem hiçbir hücreyi bölmez ve birleştirmez. Hatası, kare başına ortalama yaklaşık 3 sönük (odak dışı) çekirdeğin kaçırılmasından gelir. Baseline'ın düşük görünen Bias değeri ise 158 kaçırılan hücre ile 124 fazladan parçanın birbirini götürmesinin sonucudur.

## Veri seti

Cell Tracking Challenge – **Fluo-N2DH-GOWT1** (eğitim arşivi; resmî test arşivi **kullanılmamıştır**)

- Veri seti sayfası: https://celltrackingchallenge.net/2d-datasets/
- İndirme: https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip

Veri seti bu repoda **bulunmaz**. Zip dosyasını indirip açın ve şu yapıyı oluşturun:

```
data/Fluo-N2DH-GOWT1/
├── 01/            t000.tif ... t091.tif
├── 01_GT/TRA/     man_track000.tif ... man_track091.tif, man_track.txt
├── 02/            t000.tif ... t091.tif
└── 02_GT/TRA/     man_track000.tif ... man_track091.tif, man_track.txt
```

## Kurulum ve çalıştırma

```bash
pip install -r requirements.txt
jupyter notebook main.ipynb      # bütün adımları sırayla çalıştırır
```

`main.ipynb`, kilitli aşamaları yeniden seçmez. Aynı kodu tekrar çalıştırıp sonuçların kayıtlı dosyalarla **birebir aynı** olduğunu doğrular: veri seçimi, deney metrikleri, doğrulama tahminleri ve 15 test tahmini.

Tek tek betikler (`src/` içinden, repo kökünde çalıştırılır):

| Adım | Komut | Ödev bölümü |
|---|---|---|
| Veri seçimi | `python src/select_frames.py --data-root data/Fluo-N2DH-GOWT1` | 3 |
| Veri incelemesi | `python src/explore_data.py` | 2 |
| Referans sayılar (geliştirme+doğrulama) | `python src/reference_counts.py` | 4 |
| Baseline | `python src/baseline.py` | 5 |
| Deneyler | `python src/run_experiments.py` | 6–10 |
| Deney şekilleri | `python src/make_experiment_figures.py` | 17 |
| Doğrulama ve seçim | `python src/validate.py` | 11 |
| Test tahminlerini kilitleme | `python src/lock_test_predictions.py` | 12 |
| Test referansları | `python src/reference_counts.py --include-test` | 4, 13 |
| Test değerlendirmesi | `python src/evaluate_test.py` | 13 |
| Zamansal analiz | `python src/temporal_analysis.py` | 14 |

## Tekrarlanabilir veri seçimi

- **Seed: 31092** = öğrenci numarasının son 5 hanesi (29066) + 2026
- Geliştirme: Sequence 01'den 10 kare → 1, 4, 9, 17, 21, 27, 28, 35, 45, 91
- Doğrulama: Sequence 01'den 5 kare → 53, 59, 63, 71, 82
- Test: Sequence 02'den 3 × 5 ardışık kare → 23–27, 42–46, 76–80
- Geliştirme ile doğrulama arasında ve test blokları arasında en az 5 kare boşluk vardır. Kural sağlanmadığında aynı seed'le deterministik yeniden çekim yapılır (test seçimi 2. denemede sağlandı).

## Test sızıntısını önleme

1. Veri seçimi algoritma geliştirilmeden önce yapıldı ve commit'lendi.
2. `reference_counts.py`, `test_predictions_locked.csv` oluşmadan test referanslarını okumayı **reddeder**.
3. Bütün deneyler yalnızca 10 geliştirme karesinde yapıldı; aday seçimi 5 doğrulama karesinde yapıldı.
4. `final_pipeline.json`, `baseline_config.json` ve `test_predictions_locked.csv` oluşturulduktan sonra ilgili betikler yeniden çalışmayı reddeder.
5. Kilitli tahmin dosyası, `final_pipeline.json`'ın SHA-256 özetini içerir.

GitHub commit sırası:

| Commit | İçerik |
|---|---|
| `34dba2a` | Veri seçimi (algoritmadan önce) |
| `a9fb4d3` | Veri incelemesi, geliştirme/doğrulama referansları |
| `652fc6e` | Baseline |
| `1522f24` | Kontrollü deneyler, aday pipeline'lar |
| `9a38424` | Doğrulama, nihai pipeline sabitlendi |
| **`4c4ce28`** | **Test tahminleri kilitlendi (referanslara erişilmeden)** |
| **`2c8e849`** | **Test değerlendirmesi** |
| `447cd74` | Zamansal analiz |

## Repo yapısı

```
├── main.ipynb                     uçtan uca notebook
├── requirements.txt
├── selected_frames.csv            seçilen kareler (seed ile)
├── experiments.csv                118 kontrollü deney (+ bağlam tekrarları)
├── validation_results.csv         3 aday + baseline, doğrulama metrikleri
├── final_pipeline.json            nihai pipeline + baseline (sabitlenmiş)
├── test_predictions_locked.csv    kilitli test tahminleri
├── test_results.csv               kare bazında test sonuçları
├── src/                           bütün kaynak kod
├── results/                       ara sonuçlar (referans sayılar, özetler, olaylar)
└── figures/                       veri görüntüsü içermeyen grafikler
```

`report_figures/` klasörü (hücre görüntüsü içeren şekiller) ödev kuralı gereği repoya yüklenmez. Notebook çalıştırıldığında yeniden üretilir.

## Yöntem ve deney tasarımı (kısa)

- **Ölçütler:** MAE, RMSE, Bias (tahmin − referans ortalaması), tam doğru sayım oranı, ortalama işlem süresi.
- **Nesne düzeyinde hata ayrıştırması:** Her referans hücre ve tahmin bileşeni için kaçırılan, bölünen, birleşen ve yanlış pozitif sayıları ayrıca raporlanır (`src/metrics.py`). Bu ayrıştırma, sayım hatalarının birbirini götürdüğü durumları ortaya çıkarır.
- **Deneyler:** Aşamalı tek-faktör tasarım. Sırasıyla yoğunluk, filtre, segmentasyon, morfoloji, watershed ve minimum alan iki tur boyunca (koordinat iniş) değiştirildi. Ardından Otsu, adaptif ve sabit eşik aileleri kendi bağlamlarında tarandı.
- **Seçim kuralı:** En düşük MAE; eşitlikte RMSE, sonra süre, sonra adım sayısı.

## Kullanılan kütüphaneler

Python 3.13 · NumPy 2.5 · pandas 3.0 · SciPy 1.18 · scikit-image 0.26 · OpenCV 5.0 · Matplotlib 3.11 · tifffile
