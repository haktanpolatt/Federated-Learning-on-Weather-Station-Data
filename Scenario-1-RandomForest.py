import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

def parse_mixed_wind_speed(value):
    if isinstance(value, (int, float)): # Eğer zaten sayısal ise doğrudan döndür
        return float(value)
    if isinstance(value, str):
        # Önce "X.AyAdı" formatını dene
        match_month = re.match(r"(\d+(?:\.\d+)?)\.(?:Oca|Şub|Mar|Nis|May|Haz|Tem|Ağu|Eyl|Eki|Kas|Ara)", value, re.IGNORECASE)
        if match_month:
            try:
                return float(match_month.group(1))
            except ValueError:
                return np.nan # Sayısal kısım parse edilemezse NaN

        # Sonra doğrudan sayısal olup olmadığını dene (virgül ve nokta ondalık ayırıcılarını destekle)
        try:
            # Önce virgülü noktaya çevir (eğer varsa), sonra float'a çevirmeyi dene
            return float(value.replace(',', '.'))
        except ValueError:
            return np.nan # Hiçbir format uyuşmuyorsa NaN
    return np.nan # Diğer tipler için (list, dict vb.) NaN

try:
    df_main = pd.read_csv('Data/dataset_synop.csv', sep=';')
    # Sütun adlarındaki başındaki/sonundaki boşlukları temizle
    df_main.columns = df_main.columns.str.strip()
except FileNotFoundError:
    print("Hata: 'dataset_synop.csv' dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
    exit()

print("Veri başarıyla yüklendi.")
print("CSV'den yüklenen ve temizlenen sütun adları:", df_main.columns.tolist()) # Kontrol için eklenebilir

df_processed = df_main.copy()

# Açısal veriyi dönüştür (0-360 derece → sin/cos)
if '10-min mean wind direction' in df_processed.columns:
    radians = np.radians(df_processed['10-min mean wind direction'])
    df_processed['wind_dir_sin'] = np.sin(radians)
    df_processed['wind_dir_cos'] = np.cos(radians)
    print("wind_dir_sin ve wind_dir_cos sütunları eklendi.")

if 'wind_dir_sin' in df_processed.columns and 'Sea level pressure' in df_processed.columns:
    df_processed['wind_pressure_interaction'] = df_processed['wind_dir_sin'] * df_processed['Sea level pressure']
    print("wind_pressure_interaction sütunu eklendi.")

targets_to_predict = ['10-min mean wind speed', 'Humidity', 'Temperature', '10-min mean wind direction']
other_potential_features = [
    'Sea level pressure', '3-hour pressure variation', 'Dew point',
    'Horizontal visibility', 'Total cloud cover', 'Station pressure',
    '24-hour pressure variation', 'Precipitation in the last 24 hours'
]
engineered_features = ['wind_dir_sin', 'wind_dir_cos', 'wind_pressure_interaction']
all_relevant_columns = sorted(list(set(targets_to_predict + other_potential_features + engineered_features)))
month_string_columns = ['10-min mean wind speed'] # Bu artık yanıltıcı olabilir, belki mixed_format_wind_columns gibi bir isim?

# Sütunları işle (tip dönüşümü)
for col in df_processed.columns: # Veya sadece all_relevant_columns içinde dönebilirsiniz
    if col == '10-min mean wind speed': # Sadece bu özel sütun için yeni fonksiyonu kullan
        df_processed[col] = df_processed[col].apply(parse_mixed_wind_speed)
    elif col in all_relevant_columns and col not in month_string_columns: # Diğer sayısal sütunlar
        if col not in df_processed.columns: # Bu kontrol aslında döngü df_processed.columns üzerinde olduğu için gereksiz
            continue
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    # month_string_columns içindeki diğer sütunlar (eğer varsa) eski parse_month_string_value ile işlenebilir
    # veya onlar için de ayrı bir mantık gerekebilir. Şu an sadece rüzgar hızı için bu özel durumu ele alıyoruz.

for col in all_relevant_columns:
    if col in df_processed.columns:
        if col not in month_string_columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    else:
        # Eğer month_string_columns içinde değilse ve df_processed'da yoksa, burada da uyarı verilebilir.
        if col not in month_string_columns: # month_string_columns için uyarı yukarıda verildi
             print(f"Uyarı: Ana DataFrame'de '{col}' sütunu bulunamadı. CSV başlıklarını kontrol edin.")


station_ids = df_processed['WMO Station ID'].unique()
print(f"\n--- SENARYO 1 BAŞLIYOR (Belirtilen hedefler için Random Forest) ---")
print(f"Toplam {len(station_ids)} istasyon bulundu.")

for station_id in station_ids:
    print(f"\n--- İstasyon ID: {station_id} ---")
    
    df_station_raw = df_processed[df_processed['WMO Station ID'] == station_id].copy()
    
    station_cols_available = [col for col in all_relevant_columns if col in df_station_raw.columns]
    if not station_cols_available:
        print(f"  Uyarı: {station_id} istasyonu için {all_relevant_columns} listesinden hiçbir ilgili sütun bulunamadı. Atlanıyor.")
        continue
        
    df_station_data = df_station_raw[station_cols_available].copy()
    df_station_data.dropna(axis=1, how='all', inplace=True)
    
    if df_station_data.empty or len(df_station_data.columns) < 2:
        print(f"  Uyarı: {station_id} istasyonu için (NaN sütunlar çıkarıldıktan sonra) modelleme yapılacak yeterli ({len(df_station_data.columns)}) sütun bulunamadı. Atlanıyor.")
        continue
        
    for target_col in targets_to_predict:
        if target_col not in df_station_data.columns:
            print(f"  Uyarı: Hedef değişken '{target_col}', {station_id} istasyon verisinde (df_station_data içinde) bulunmuyor. Atlanıyor.")
            continue
        
        print(f"  --- Hedef Değişken: {target_col} ---")
        
        y_station_full = df_station_data[target_col].copy()
        X_station_full = df_station_data.drop(columns=[target_col], errors='ignore')

        if X_station_full.empty:
            print(f"    Uyarı: {target_col} için özellik kalmadı. Atlanıyor.")
            continue
            
        valid_target_indices = y_station_full.dropna().index
        y_station = y_station_full.loc[valid_target_indices]
        X_station = X_station_full.loc[valid_target_indices]
        
        if X_station.empty or y_station.empty or len(X_station) < 5:
            print(f"    Uyarı: {target_col} için hedefte NaN olmayan yeterli veri ({len(X_station)} satır) kalmadı. Atlanıyor.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(X_station, y_station, test_size=0.2, random_state=42)
        
        if X_train.empty or X_test.empty:
            print(f"    Uyarı: {target_col} için train/test split sonrası boş küme(ler) oluştu. Atlanıyor.")
            continue

        feature_names = X_train.columns.tolist()
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_names)
        X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=feature_names)

        y_test_reset = y_test.reset_index(drop=True)
        X_test_imputed_reset = X_test_imputed.reset_index(drop=True)

        y_test_name_original = y_test_reset.name if y_test_reset.name is not None else target_col # Fallback to target_col
        y_test_name_for_concat = y_test_name_original
        # Ensure unique column name for y in concat
        i = 0
        while y_test_name_for_concat in X_test_imputed_reset.columns:
            y_test_name_for_concat = f"{y_test_name_original}_{i}"
            i += 1
        if y_test_name_original != y_test_name_for_concat:
             y_test_reset = y_test_reset.rename(y_test_name_for_concat)


        temp_test_df = pd.concat([X_test_imputed_reset, y_test_reset], axis=1)
        cleaned_test_df = temp_test_df.dropna()

        if cleaned_test_df.empty:
            print(f"    Uyarı: {target_col} için test setinde NaN temizliği sonrası veri kalmadı. Atlanıyor.")
            continue
            
        X_test_final = cleaned_test_df.drop(columns=[y_test_name_for_concat]) # Use the name used in concat
        y_test_final = cleaned_test_df[y_test_name_for_concat]
        
        if X_test_final.empty or y_test_final.empty or len(X_test_final) < 1 :
            print(f"    Uyarı: {target_col} için test seti temizlik sonrası yetersiz ({len(X_test_final)} örnek). Atlanıyor.")
            continue

        model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=None) 
        
        try:
            model.fit(X_train_imputed, y_train)
            y_pred = model.predict(X_test_final)
            
            mae = mean_absolute_error(y_test_final, y_pred)
            mse = mean_squared_error(y_test_final, y_pred)
            rmse = np.sqrt(mse)
            r2 = np.nan
            mape = np.nan 
            r2_note = ""

            if len(y_test_final) >= 2:
                r2 = r2_score(y_test_final, y_pred)
            else:
                r2_note = " (R² tanımsız: <2 test örneği)"
            
            # Calculate MAPE if possible
            # Check for zeros in y_test_final to avoid division by zero or انفجار MAPE
            if not np.isinf(y_pred).any() and not np.isnan(y_pred).any(): # Ensure y_pred is clean
                if np.all(y_test_final != 0): # Ensure no zeros in actual values
                    mape = mean_absolute_percentage_error(y_test_final, y_pred) * 100
                else:
                    # For rows where y_test_final is 0, MAPE is problematic.
                    # We can calculate MAPE for non-zero rows or report as N/A.
                    non_zero_mask = y_test_final != 0
                    if np.any(non_zero_mask): # If there are some non-zero values
                        mape = mean_absolute_percentage_error(y_test_final[non_zero_mask], y_pred[non_zero_mask]) * 100
                        print(f"      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.")
                    else: # All y_test_final are zero
                        print(f"      Bilgi: MAPE hesaplanamıyor (tüm y_test_final değerleri sıfır).")
                        mape = np.nan # explicitly set to nan
            else:
                print(f"      Bilgi: MAPE hesaplanamıyor (y_pred'de inf veya NaN değerler var).")


            # Prepare MAPE string for printing
            mape_str_display = "N/A"
            if not np.isnan(mape) and not np.isinf(mape):
                mape_str_display = f"{mape:.2f}%"
            
            r2_display = f"{r2:.4f}" if not np.isnan(r2) else "N/A" 
            print(f"    Random Forest: MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2_display}{r2_note}, MAPE={mape_str_display}")

        except Exception as e:
            print(f"    Hata (Random Forest modeli, {target_col} hedefi için, İstasyon {station_id}): {e}")
            import traceback
            traceback.print_exc()


print("\nSenaryo 1 tamamlandı.")