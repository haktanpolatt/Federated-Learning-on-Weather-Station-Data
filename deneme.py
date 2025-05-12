import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Veriyi oku
df = pd.read_csv("dataset_synop.csv")

# Zaman damgasını datetime'a çevir
df["Date"] = pd.to_datetime(df["Date"])

# Ortak fiziksel büyüklükleri seçelim (örnek olarak)
common_features = ["Temperature", "Dew point", "Humidity", "Sea level pressure"]
target = "Temperature"  # tahmin edilecek değer

# Sayısal dönüşüm (bazı sütunlar nokta yerine virgül içerebilir)
df[common_features] = df[common_features].apply(pd.to_numeric, errors='coerce')

# Tarihe göre sırala (zaman serisi uyumlu hale getirmek için)
df = df.sort_values("Date")

# NaN’leri düşürmek yerine, pipeline ile dolduracağız
X = df[common_features]
y = df[target]

# Pipeline tanımı
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(random_state=42))
])

# GridSearch ile hiperparametre ayarı
param_grid = {
    'model__n_estimators': [50, 100],
    'model__max_depth': [5, 10, None],
    'model__min_samples_split': [2, 5]
}

# Zaman serisine uygun bölme
tscv = TimeSeriesSplit(n_splits=5)

grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='r2', verbose=1, n_jobs=-1)
grid_search.fit(X, y)

# En iyi modeli kaydet
joblib.dump(grid_search.best_estimator_, "best_rf_model.pkl")

# Performans değerlendirmesi
y_pred = grid_search.predict(X)
print("R2 Score:", r2_score(y, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y, y_pred)))
