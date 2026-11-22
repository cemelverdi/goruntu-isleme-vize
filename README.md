# Klasik Görüntü İşleme ile Mikroskobik Hücre Segmentasyonu, Sayımı ve Zamansal Analizi

Görüntü İşleme (Yüksek Lisans) ara sınav projesi.

> Bu README proje ilerledikçe tamamlanacaktır.

## Veri seti

Cell Tracking Challenge – **Fluo-N2DH-GOWT1** (eğitim arşivi)

- Veri seti sayfası: https://celltrackingchallenge.net/2d-datasets/
- İndirme: https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip

Veri seti bu repoda **bulunmaz**. Zip'i indirip açın ve aşağıdaki yapıyı oluşturun:

```
data/Fluo-N2DH-GOWT1/
├── 01/            t000.tif ... t091.tif
├── 01_GT/TRA/     man_track000.tif ... man_track091.tif
├── 02/
└── 02_GT/TRA/
```

## Tekrarlanabilir veri seçimi

- Seed: **31092** = öğrenci numarasının son 5 hanesi (29066) + 2026
- Geliştirme: 10 kare (Sequence 01), Doğrulama: 5 kare (Sequence 01), Test: 3 × 5 ardışık kare (Sequence 02)

```bash
pip install -r requirements.txt
python src/select_frames.py --data-root data/Fluo-N2DH-GOWT1 --out selected_frames.csv
```

Seçim, algoritma geliştirilmeden önce yapılmış ve `selected_frames.csv` dosyasına kaydedilmiştir.
