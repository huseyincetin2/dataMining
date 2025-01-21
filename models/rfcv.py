from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import numpy as np

# RMSE scorer oluşturma
rmse_scorer = make_scorer(lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)))

# K-Fold Cross Validation
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
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

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

# Learning Curve grafiği
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