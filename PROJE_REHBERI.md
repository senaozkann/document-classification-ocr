# Proje Rehberi — Dosyalar Nerede?

Bu klasör projenin **kodunu** içerir. Proje Google Colab'da (bulutta) çalışır; veriler Google Drive'dadır.

| Parça | Nerede | İçerik |
|---|---|---|
| Kod (bu klasör) | Masaüstü + GitHub: `senaozkann/document-classification-ocr` | Notebook'lar, `src/` modülleri, README |
| Veriler ve sonuçlar | Google Drive → `MyDrive/bitirme_ocr/` | 2.400 belge görüntüsü, OCR çıktıları, model, grafikler |
| Çalıştırma | Google Colab | Notebook'lar burada çalıştırılır |

## Klasör yapısı

```
document-classification-ocr/
├── README.md                 Proje özeti ve sonuçlar
├── PROJE_REHBERI.md          Bu dosya
├── notebooks/
│   ├── 01_data_preparation.ipynb       Veri setini indir (RVL-CDIP, 4 sınıf)
│   ├── 02_ocr_pipeline.ipynb           OCR ile metin çıkar
│   ├── 03_ocr_analysis_cleaning.ipynb  OCR analizi, metin temizleme
│   ├── 04_classification.ipynb         TF-IDF + NB / LR / SVM, metrikler
│   ├── 05_end_to_end_demo.ipynb        Yeni belge → tür + içerik, Gradio demo
│   ├── 06_sinif_secimi.ipynb           Form yerine hangi sınıf? Aday sınıf deneyi
│   ├── 07_sablon_ve_sozluk.ipynb       Anahtar kelime şablonu + sözlükle OCR düzeltme
│   ├── 08_gurultu_giderme.ipynb        Gürültü giderme yöntemlerinin karşılaştırması
│   ├── 09_elle_degerlendirme.ipynb     İçerik çıkarımını resme bakarak elle değerlendirme
│   └── 10_son_model.ipynb              Son model (metin + tablo), "emin değilim", çıkarım önce/sonra
└── src/
    ├── ocr_core.py           OCR fonksiyonları
    ├── text_clean.py         Metin temizleme
    ├── extract.py            Türe özel içerik çıkarımı
    ├── sablon.py             Belge türü şablonları
    └── ocr_fix.py            Sözlükle OCR düzeltme
```

## Drive'daki yapı (`MyDrive/bitirme_ocr/`)

```
bitirme_ocr/
├── data/
│   ├── images/               Belge görüntüleri (train / validation / test)
│   ├── ocr_raw/              OCR sonuçları (v2, kullanılan)
│   ├── metadata.csv          Belge listesi + gerçek tür
│   ├── ocr_dataset.csv       OCR metinleri
│   └── cleaned_dataset.csv   Temizlenmiş metinler
├── models/                   Eğitilmiş model
├── figures/                  Rapor grafikleri
└── kendi_belgelerim/         Demo için kendi belgeleriniz
```

## Çalıştırma

1. colab.research.google.com → **File → Upload notebook** → `notebooks/` içinden dosyayı seç
2. **Runtime → Run all**
3. Sıra: 01 → 02 → 06 → 03 → 04 → 10 → 05 → 07 → 08 (10, 05'ten önce çalışırsa demo son modeli kullanır)
