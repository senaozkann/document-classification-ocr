# İçerik çıkarımının yeni 20 belgede değerlendirilmesi (22 Ekim 2026 · notebook 12)

Kurallar (extract.py v2) **değiştirilmeden**, test setinden daha önce hiç bakılmamış 20 yeni belge (her türden 5, `random_state=2026`)
tam sistemle (yön düzeltme + gerekirse büyütme → OCR → etiket–değer birleştirme → çıkarım) işlendi. 110 alan elle işaretlendi:
Doğru (d) · Yanlış (y) · Kaçırıldı (k) · Belgede yok (b). Küçük OCR harf hataları doğru sayıldı.

| Tür | d | y | k | b | Precision | Recall | Doğruluk |
|---|---|---|---|---|---|---|---|
| E-posta | 15 | 3 | 3 | 9 | 0,83 | 0,83 | 0,80 |
| Fatura | 7 | 5 | 8 | 5 | 0,58 | 0,47 | 0,48 |
| CV | 6 | 8 | 0 | 16 | 0,43 | 1,00 | 0,73 |
| Bilimsel makale | 5 | 5 | 3 | 12 | 0,50 | 0,63 | 0,68 |
| **Toplam** | **33** | **21** | **14** | **42** | **0,61** | **0,70** | **0,68** |

## Eski 20 belgeyle karşılaştırma

| | Precision | Recall | Doğruluk |
|---|---|---|---|
| Eski 20, kurallar v1 (09) | 0,525 | 0,744 | 0,636 |
| Eski 20, kurallar v2 (kurallar bu belgelere bakılarak düzeltildi) | 0,897 | 0,825 | 0,845 |
| **Yeni 20, kurallar v2 (görülmemiş belgeler)** | **0,611** | **0,702** | **0,682** |

Eski belgelerdeki 0,90 iyimserdi (aşırı uyum); görülmemiş belgelerde precision 0,61. v1'e göre yine de daha iyi (0,525 → 0,611),
yani düzeltmelerin bir kısmı genelleşiyor.

## Başlıca hata kaynakları

- **Etiket gürültüsü:** 3 belge türüne uymuyor (2: sigara spesifikasyon tablosu "email", 10: posta yönlendirme fişi "invoice",
  11: danışma kurulu listesi "resume"). Bunlar hariç: precision 0,66, recall 0,70.
- **Fatura firması:** firma adı çoğunlukla logo (resim) içinde; OCR okuyamıyor, sistem adres/slogan satırını alıyor (5'te 4 yanlış).
- **Fatura no / toplam:** "NO." ya da voucher numarası ve "Pay This Amount" gibi etiketler kural listesinde yok.
- **CV bölüm/derece:** eğitim bilgisi tablo sütunlarında (DEGREE / FIELD OF STUDY); satır tabanlı kural sütun başlığını ya da ödül satırını alıyor.
- **Makale özeti:** "Abstract" başlığı olmayan ya da sütun içinde kalan özetler kaçırılıyor.

## Sınırda kararlar

Belge 1 gönderen ("gan, Karen M." → doğrusu "Daragan, Karen M.", d sayıldı) · 12 bölüm ("1974 B.A cum laude…" ödül satırı, y) ·
13 üniversite (onur satırından "Dartmouth College", d) · 16 başlık (son satır eksik, y) · 19 yazarlar (6 yazardan 5'i, d) · 20 yazarlar (5'ten 2'si, y).

| Dosya | İçerik |
|---|---|
| `elle_kararlar.txt` | Belge başına kararlar (alan sırası notebook 12 ile aynı) |
| `cikarim_degerlendirme_yeni20.csv` | 110 alan: belge, tür, alan, sistemin değeri, karar |
| `cikarim_sonuclari_tur_yeni20.csv` · `cikarim_sonuclari_alan_yeni20.csv` | Tür / alan bazında sonuçlar |
| `cikarim_inceleme_yeni20.pdf` | Notebook 12'nin inceleme PDF'i |
