import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

# --- Veri Yükleme ve Ön İşleme Fonksiyonları (senaryo2.py'den benzer) ---

def parse_month_string_value(value):
    if isinstance(value, str):
        match = re.match(r"(\d+)\.(?:Oca|Şub|Mar|Nis|May|Haz|Tem|Ağu|Eyl|Eki|Kas|Ara)", value, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return np.nan
    return np.nan

try:
    df_main = pd.read_csv('dataset_synop.csv', sep=';')
except FileNotFoundError:
    print("Hata: 'dataset_synop.csv' dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
    exit()

print("Veri başarıyla yüklendi.")

df_processed = df_main.copy()

# Modellemede kullanılacak temel fiziksel ölçüm sütunları
physical_measurement_columns = [
    'Sea level pressure',
    '3-hour pressure variation',
    # '10-min mean wind speed', # Bu sütun bozuk olduğu için çıkarıldı, gerekirse eklenebilir
    'Temperature',
    'Dew point',
    'Humidity',
    'Horizontal visibility',
    'Total cloud cover',
    'Station pressure',
    '24-hour pressure variation',
    'Precipitation in the last 24 hours'
]

# "X.AyAdı" formatında parse edilecek sütunlar (eğer varsa ve kullanılacaksa)
# '10-min mean wind speed' bozuk olduğu için şimdilik boş bırakıyorum.
month_string_columns = [] # Örn: ['10-min mean wind speed']

# 1. "X.AyAdı" formatındaki sütunları işle (eğer varsa)
for col in month_string_columns:
    if col in df_processed.columns:
        df_processed[col] = df_processed[col].apply(parse_month_string_value)

# 2. Seçilen tüm fiziksel ölçüm sütunlarını sayısal değere dönüştür, hataları NaN yap
for col in physical_measurement_columns:
    if col in df_processed.columns:
        if col not in month_string_columns or pd.api.types.is_object_dtype(df_processed[col]):
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    else:
        print(f"Uyarı: Ana DataFrame'de {col} sütunu bulunamadı.")

# --- Senaryo 1: Her istasyon için lokal verilerle Random Forest ---

station_ids = df_processed['WMO Station ID'].unique()

print(f"\n--- SENARYO 1 BAŞLIYOR (Sadece Random Forest) ---")
print(f"Toplam {len(station_ids)} istasyon bulundu: {station_ids}")

for station_id in station_ids:
    print(f"\n--- İstasyon ID: {station_id} ---")
    
    df_station = df_processed[df_processed['WMO Station ID'] == station_id].copy()
    
    # Bu istasyon için modellemede kullanılacak sütunları belirle
    # Sadece bu istasyonda var olan ve physical_measurement_columns listesindeki sütunları al
    station_physical_cols_present = [col for col in physical_measurement_columns if col in df_station.columns]
    df_station_model_data = df_station[station_physical_cols_present].copy()
    
    # Bu istasyon için tamamen NaN olan sütunları çıkar
    df_station_model_data.dropna(axis=1, how='all', inplace=True)
    
    # Modelleme için kullanılacak nihai sütun listesi (bu istasyon için)
    station_cols_for_modeling = df_station_model_data.columns.tolist()
    
    if not station_cols_for_modeling or len(station_cols_for_modeling) < 2: # En az 1 özellik 1 hedef lazım
        print(f"  Uyarı: {station_id} istasyonu için modelleme yapılacak yeterli ({len(station_cols_for_modeling)}) sütun bulunamadı. Atlanıyor.")
        continue
        
    # Eksik değerleri bu istasyonun verileriyle medyan ile doldur
    imputer = SimpleImputer(strategy='median')
    df_station_imputed = pd.DataFrame(imputer.fit_transform(df_station_model_data), columns=station_cols_for_modeling)
    
    if df_station_imputed.empty or len(df_station_imputed) < 5: # Train/test split için yeterli satır kontrolü
        print(f"  Uyarı: {station_id} istasyonu için eksik değer doldurma sonrası yeterli veri ({len(df_station_imputed)} satır) kalmadı. Atlanıyor.")
        continue

    for target_col in station_cols_for_modeling:
        print(f"  --- Hedef Değişken: {target_col} ---")
        
        y_station = df_station_imputed[target_col]
        X_station = df_station_imputed.drop(columns=[target_col])
        
        if X_station.empty:
            print(f"    Uyarı: {target_col} için özellik kalmadı. Atlanıyor.")
            continue
        
        # Veriyi eğitim ve test setlerine ayır (istasyon özelinde)
        # Yeterli veri yoksa (örn: <2 sample testte) hata verebilir.
        # test_size=0.2, en az 5 satır olmalı ki testte 1 satır kalsın.
        # R2 için testte en az 2 satır olması daha iyi.
        min_samples_for_split = 5 
        if len(X_station) < min_samples_for_split:
            print(f"    Uyarı: {target_col} için yetersiz veri ({len(X_station)} satır). Train/test split yapılamıyor. Atlanıyor.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(X_station, y_station, test_size=0.2, random_state=42)
        
        if X_test.empty:
            print(f"    Uyarı: {target_col} için test seti boş kaldı. Atlanıyor.")
            continue

        # Random Forest Model
        # Parametreleri veri setinin büyüklüğüne göre ayarlayabilirsin.
        # Örnek: n_estimators=100, max_depth=None (veya daha büyük bir sayı)
        model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=None) 
        
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = np.nan
            mape = np.nan
            r2_note = ""

            if len(y_test) >= 2: # R2 için en az 2 örnek
                r2 = r2_score(y_test, y_pred)
            else:
                r2_note = " (R² tanımsız: <2 örnek)"
            
            try:
                # y_test'te 0 varsa MAPE tanımsız olabilir.
                # np.finfo(float).eps küçük bir sayı ekleyerek sıfıra bölmeyi engelleme denemesi (dikkatli kullanılmalı)
                # Veya sklearn'in kendi MAPE'si genellikle bunu yönetir.
                mape = mean_absolute_percentage_error(y_test, y_pred) * 100 # Yüzde olarak
            except ZeroDivisionError: # Bu aslında sklearn.metrics.mean_absolute_percentage_error'da pek olmaz
                 print(f"      UYARI: MAPE hesaplanırken sıfıra bölme (y_test'te 0 olabilir). MAPE NaN.")
            
            print(f"    Random Forest: MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}{r2_note}, MAPE={mape:.2f}%")

        except Exception as e:
            print(f"    Hata (Random Forest modeli, {target_col} hedefi için): {e}")

print("\nSenaryo 1 tamamlandı.")