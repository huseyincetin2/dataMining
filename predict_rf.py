import pandas as pd
import numpy as np
from joblib import load
import warnings
warnings.filterwarnings('ignore')

def load_models():
    try:
        # Model ve encoder'ları yükle
        model = load('models/random_forest_model2.joblib')
        encoders = load('models/label_encoders2.joblib')
        return model, encoders
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        return None, None

def get_student_history(student_id):
    try:
        # Veri setini yükle
        df = pd.read_csv('dataset/data2.csv', encoding='utf-8')
        # Öğrenci verilerini filtrele
        student_data = df[df['emsStudentId'] == student_id]
        if student_data.empty:
            print(f"ID: {student_id} olan öğrenci bulunamadı!")
            return None
        return student_data, df
    except Exception as e:
        print(f"Veri okuma hatası: {e}")
        return None

def predict_student_success(student_id):
    # Model ve veriyi yükle
    model, encoders = load_models()
    result = get_student_history(student_id)
    
    if model is None or result is None:
        return None
        
    student_data, df = result
    
    try:
        # Öğrencinin ders bazında performans metriklerini hesapla
        subject_metrics = student_data.groupby('lessonName').agg({
            'SS': 'mean',
            'DS': 'mean',
            'YS': 'mean',
            'Net': 'mean',
            'Yuzde': 'mean'
        }).to_dict('index')
        
        # Mevcut başarılı dersleri bul
        strong_subjects = [subject for subject, metrics in subject_metrics.items() 
                         if metrics['Yuzde'] >= 70]
        weak_subjects = [subject for subject, metrics in subject_metrics.items() 
                        if metrics['Yuzde'] < 50]
        
        # Genel ortalama metrikleri (yeni dersler için)
        general_metrics = {
            'SS': student_data['SS'].mean(),
            'DS': student_data['DS'].mean(),
            'YS': student_data['YS'].mean(),
            'Net': student_data['Net'].mean(),
            'current_success': student_data['Yuzde'].mean()
        }
        
        # Tüm dersler için tahmin yap
        predictions = {}
        all_lessons = df['lessonName'].unique()
        
        for lesson in all_lessons:
            try:
                # Dersin örnek konu ve çıktısını al
                lesson_data = df[df['lessonName'] == lesson]
                if not lesson_data.empty:
                    lesson_example = lesson_data.iloc[0]
                    
                    # Eğer öğrencinin bu dersten geçmişi varsa, o dersin metriklerini kullan
                    if lesson in subject_metrics:
                        metrics = subject_metrics[lesson]
                        ss = metrics['SS']
                        ds = metrics['DS']
                        ys = metrics['YS']
                        net = metrics['Net']
                    else:
                        # Yoksa genel ortalama metrikleri kullan
                        ss = general_metrics['SS']
                        ds = general_metrics['DS']
                        ys = general_metrics['YS']
                        net = general_metrics['Net']
                    
                    # Tahmin verisi oluştur
                    prediction_data = pd.DataFrame({
                        'SS': [ss],
                        'DS': [ds],
                        'YS': [ys],
                        'Net': [net],
                        'lessonName_encoded': [encoders['lessonName'].transform([lesson])[0]],
                        'topicName_encoded': [encoders['topicName'].transform([lesson_example['topicName']])[0]],
                        'outcomeName_encoded': [encoders['outcomeName'].transform([lesson_example['outcomeName']])[0]]
                    })
                    
                    # Tahmin yap
                    pred = model.predict(prediction_data)[0]
                    
                    # Eğer öğrencinin bu dersten geçmişi varsa, mevcut başarısını da dikkate al
                    if lesson in subject_metrics:
                        current_success = subject_metrics[lesson]['Yuzde']
                        # Mevcut başarı ve tahmini başarının ağırlıklı ortalaması
                        pred = (0.7 * pred + 0.3 * current_success)
                    
                    predictions[lesson] = pred
                    
            except Exception as e:
                print(f"{lesson} dersi için tahmin yapılamadı: {e}")
                continue
        
        return {
            'student_id': student_id,
            'current_metrics': general_metrics,
            'subject_metrics': subject_metrics,
            'strong_subjects': strong_subjects,
            'weak_subjects': weak_subjects,
            'predictions': predictions
        }
        
    except Exception as e:
        print(f"Tahmin hatası: {e}")
        return None

def print_predictions(results):
    if results is None:
        return
    
    print("\n🎓 ÖĞRENCİ BAŞARI TAHMİN RAPORU")
    print("=" * 50)
    
    print(f"\n📊 Öğrenci ID: {results['student_id']}")
    print(f"Mevcut Başarı Ortalaması: %{results['current_metrics']['current_success']:.1f}")
    
    print("\n💪 Mevcut Güçlü Dersler:")
    for subject in results['strong_subjects']:
        print(f"• {subject}")
    
    print("\n📚 Geliştirilmesi Gereken Dersler:")
    for subject in results['weak_subjects']:
        print(f"• {subject}")
    
    print("\n🔮 Gelecek Başarı Tahminleri:")
    predictions = results['predictions']
    sorted_predictions = dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True))
    
    print("\nYüksek Potansiyel Gösteren Dersler (>= 70):")
    high_potential = {k: v for k, v in sorted_predictions.items() if v >= 70}
    for subject, pred in high_potential.items():
        print(f"• {subject}: %{pred:.1f}")
    
    print("\nOrta Potansiyel Gösteren Dersler (50-70):")
    mid_potential = {k: v for k, v in sorted_predictions.items() if 50 <= v < 70}
    for subject, pred in mid_potential.items():
        print(f"• {subject}: %{pred:.1f}")
    
    print("\nDüşük Potansiyel Gösteren Dersler (< 50):")
    low_potential = {k: v for k, v in sorted_predictions.items() if v < 50}
    for subject, pred in low_potential.items():
        print(f"• {subject}: %{pred:.1f}")
    
    # Öneriler
    print("\n📝 ÖNERİLER:")
    if high_potential:
        print("• Yüksek potansiyel gösterilen derslere odaklanarak başarınızı artırabilirsiniz.")
    if low_potential:
        print("• Düşük potansiyel gösterilen dersler için ek destek alınması önerilir.")
    print("• Düzenli çalışma programı oluşturun.")
    print("• Konuları tekrar edin ve bol soru çözün.")

def main():
    try:
        student_id = int(input("\n🔍 Öğrenci ID'sini girin: "))
        results = predict_student_success(student_id)
        print_predictions(results)
    except ValueError:
        print("Hata: Geçerli bir öğrenci ID'si girin!")

if __name__ == "__main__":
    main()