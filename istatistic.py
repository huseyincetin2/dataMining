import pandas as pd

# CSV dosyasını okuma
df = pd.read_csv('dataset/data2.csv')

# İstatistikleri hesaplama
tekil_ogrenci_sayisi = df['emsStudentId'].nunique()
tekil_ders_sayisi = df['lessonName'].nunique()
tekil_topic_sayisi = df['topicName'].nunique()
tekil_outcome_sayisi = df['outcomeName'].nunique()
genel_basari_ortalamasi = df['Yuzde'].mean()
toplam_satir_sayisi = len(df)

# Sonuçları yazdırma
print("=== Veri Seti İstatistikleri ===")
print(f"Tekil Öğrenci Sayısı: {tekil_ogrenci_sayisi}")
print(f"Tekil Ders Sayısı: {tekil_ders_sayisi}")
print(f"Tekil Konu (Topic) Sayısı: {tekil_topic_sayisi}")
print(f"Tekil Kazanım (Outcome) Sayısı: {tekil_outcome_sayisi}")
print(f"Genel Başarı Ortalaması: %{genel_basari_ortalamasi:.2f}")
print(f"Toplam Satır Sayısı: {toplam_satir_sayisi}")
