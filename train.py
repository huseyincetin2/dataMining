import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer
import matplotlib.pyplot as plt
from joblib import dump
import warnings
warnings.filterwarnings('ignore')

# Veriyi okuma
df = pd.read_csv('dataset/data2.csv') 

# Temel istatistikleri görme
print("Veri seti boyutu:", df.shape)
print("\nSütunlar:", df.columns.tolist())
print("\nVeri seti örneği:")
print(df.head())

# Eksik verileri kontrol etme ve temizleme
print("\nEksik değerler:")
print(df.isnull().sum())
df = df.dropna(subset=['Yuzde'])

# Kategorik değişkenleri sayısala çevirme
le_lesson = LabelEncoder()
le_topic = LabelEncoder()
le_outcome = LabelEncoder()  # Yeni eklenen encoder

# Ders, konu ve outcome'ları ayrı ayrı encode etme
df['lessonName_encoded'] = le_lesson.fit_transform(df['lessonName'])
df['topicName_encoded'] = le_topic.fit_transform(df['topicName'])
df['outcomeName_encoded'] = le_outcome.fit_transform(df['outcomeName'])

# Özellik seçimi
features = [
    'SS', 
    'DS', 
    'YS', 
    'Net', 
    'lessonName_encoded', 
    'topicName_encoded',
    'outcomeName_encoded'
]
target = 'Yuzde'

X = df[features]
y = df[target]

# Eğitim ve test verilerini ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nEğitim veri seti boyutu:", X_train.shape)
print("Test veri seti boyutu:", X_test.shape)

# Random Forest modelini oluşturma
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

# Modeli eğitme
rf_model.fit(X_train, y_train)

# Tahminleme
y_pred = rf_model.predict(X_test)

# Model performansını değerlendirme
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\nModel Performansı:")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")
print(f"MAE: {mae:.2f}")

# Cross Validation analizi
rmse_scorer = make_scorer(lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)))
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Cross Validation skorları
cv_rmse_scores = cross_val_score(rf_model, X, y, cv=kf, scoring=rmse_scorer)
cv_r2_scores = cross_val_score(rf_model, X, y, cv=kf, scoring='r2')

print("\nCross Validation Sonuçları:")
print(f"RMSE skorları: {cv_rmse_scores}")
print(f"Ortalama RMSE: {cv_rmse_scores.mean():.2f} (+/- {cv_rmse_scores.std() * 2:.2f})")
print(f"\nR2 skorları: {cv_r2_scores}")
print(f"Ortalama R2: {cv_r2_scores.mean():.2f} (+/- {cv_r2_scores.std() * 2:.2f})")

# Train vs Validation performans karşılaştırması
train_predictions = rf_model.predict(X_train)
test_predictions = rf_model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

train_r2 = r2_score(y_train, train_predictions)
test_r2 = r2_score(y_test, test_predictions)

print("\nTrain vs Test Performans Karşılaştırması:")
print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")
print(f"Train R2: {train_r2:.2f}")
print(f"Test R2: {test_r2:.2f}")

# Overfitting kontrolü
print("\nOverfitting Analizi:")
rmse_diff = abs(train_rmse - test_rmse)
r2_diff = abs(train_r2 - test_r2)

print(f"RMSE farkı: {rmse_diff:.2f}")
print(f"R2 farkı: {r2_diff:.2f}")

if train_rmse < test_rmse and (test_rmse - train_rmse) / train_rmse > 0.2:
    print("UYARI: Model overfitting gösteriyor!")
    print("Öneriler:")
    print("1. max_depth parametresini azaltın")
    print("2. min_samples_split değerini artırın")
    print("3. min_samples_leaf değerini artırın")
    print("4. n_estimators değerini azaltın")
elif abs(train_rmse - test_rmse) / train_rmse < 0.1:
    print("Model dengeli görünüyor.")
else:
    print("Model kabul edilebilir düzeyde, ancak iyileştirilebilir.")

# Learning Curve analizi
train_sizes, train_scores, test_scores = learning_curve(
    rf_model, X, y,
    cv=5,
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='neg_mean_squared_error'
)

# RMSE'ye çevirme
train_scores_rmse = np.sqrt(-train_scores)
test_scores_rmse = np.sqrt(-test_scores)

# Ortalama ve standart sapma
train_mean = train_scores_rmse.mean(axis=1)
train_std = train_scores_rmse.std(axis=1)
test_mean = test_scores_rmse.mean(axis=1)
test_std = test_scores_rmse.std(axis=1)

# Modeli ve encoder'ları kaydetme
# models klasörünü oluştur (eğer yoksa)
import os
if not os.path.exists('models'):
    os.makedirs('models')

model_filename = 'models/random_forest_model2.joblib'
encoder_filename = 'models/label_encoders2.joblib'

# Encoder'ları ayrı ayrı kaydet
encoders = {
    'lessonName': le_lesson,
    'topicName': le_topic,
    'outcomeName': le_outcome
}

dump(rf_model, model_filename)
dump(encoders, encoder_filename)

print(f"\nModel kaydedildi: {model_filename}")
print(f"Encoder'lar kaydedildi: {encoder_filename}")

# Eğitim verisindeki benzersiz değerleri yazdır
print("\nEğitim verisindeki benzersiz dersler:")
print(df['lessonName'].unique())
print("\nEğitim verisindeki benzersiz konular:")
print(df['topicName'].unique())
print("\nEğitim verisindeki benzersiz çıktılar:")
print(df['outcomeName'].unique())

# GÖRSELLEŞTIRMELER

# 1. Özellik önemlilikleri
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.xticks(rotation=45)
plt.title('Özellik Önemlilikleri')
plt.tight_layout()
plt.show()

# 2. Gerçek vs Tahmin değerleri
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.title('Gerçek vs Tahmin Değerleri')
plt.tight_layout()
plt.show()

# 3. Learning Curve grafiği
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score')
plt.plot(train_sizes, test_mean, label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
plt.xlabel('Training Examples')
plt.ylabel('RMSE Score')
plt.title('Learning Curve')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# 4. Ders bazında başarı dağılımı
plt.figure(figsize=(12, 6))
df.groupby('lessonName')['Yuzde'].mean().sort_values(ascending=False).plot(kind='bar')
plt.title('Derslere Göre Ortalama Başarı Yüzdesi')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. Korelasyon matrisi
correlation_matrix = df[['SS', 'DS', 'YS', 'Net', 'Yuzde']].corr()
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('Özellikler Arası Korelasyon Matrisi')
plt.tight_layout()
plt.show() 