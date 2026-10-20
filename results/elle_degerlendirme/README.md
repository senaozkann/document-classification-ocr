# İçerik çıkarımının elle değerlendirilmesi (21 Ekim 2026)

Test setinden her türden 5 belge (20 belge, 110 alan). Her alan için karar:
Doğru (d) · Yanlış (y) · Kaçırıldı (k) · Belgede yok (b). Küçük OCR harf hataları doğru sayıldı.

| Dosya | İçerik |
|---|---|
| `elle_kararlar.txt` | Belge başına kararlar (belge no + harfler, alan sırası notebook 09 ile aynı) |
| `cikarim_degerlendirme_v2.csv` | 110 alanın her biri: belge, tür, alan, sistemin değeri, karar |
| `cikarim_sonuclari_tur_v2.csv` | Türe göre precision / recall / doğruluk |
| `cikarim_sonuclari_v2.csv` | Alana göre precision / recall / doğruluk |

Toplam: precision 0,525 · recall 0,744 · doğruluk 0,636.
