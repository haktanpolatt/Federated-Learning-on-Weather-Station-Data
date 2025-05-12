import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
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
    df = pd.read_csv('Data/dataset_synop.csv', sep=';')
except FileNotFoundError:
    print("Hata: 'dataset_synop.csv' dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
    exit()

print("Veri başarıyla yüklendi.")
df_processed = df.copy()

# Açısal veri dönüşümü: sin/cos bileşenleri
if '10-min mean wind direction' in df_processed.columns:
    radians = np.radians(pd.to_numeric(df_processed['10-min mean wind direction'], errors='coerce'))
    df_processed['wind_dir_sin'] = np.sin(radians)
    df_processed['wind_dir_cos'] = np.cos(radians)
    print("wind_dir_sin ve wind_dir_cos sütunları eklendi.")

if 'wind_dir_sin' in df_processed.columns and 'Sea level pressure' in df_processed.columns:
    df_processed['wind_pressure_interaction'] = df_processed['wind_dir_sin'] * df_processed['Sea level pressure']
    print("wind_pressure_interaction sütunu eklendi.")

# Hedeflenecek ve özellik olarak kullanılabilecek sütunlar
targets_to_predict = ['10-min mean wind speed', 'Humidity', 'Temperature', '10-min mean wind direction']
other_potential_features = [
    'Sea level pressure', '3-hour pressure variation', 'Dew point',
    'Horizontal visibility', 'Total cloud cover', 'Station pressure',
    '24-hour pressure variation', 'Precipitation in the last 24 hours'
    # 'Gust over last 10 minutes' # Bu sütun da parse_month_string_value gerektirebilir, eklenirse month_string_columns'a da eklenmeli
]
engineered_features = ['wind_dir_sin', 'wind_dir_cos', 'wind_pressure_interaction']
all_relevant_columns = sorted(list(set(targets_to_predict + other_potential_features + engineered_features)))

# "X.AyAdı" formatında parse edilecek sütunlar
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
        if col not in month_string_columns: # Daha önce parse edilmediyse
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    else:
        print(f"Uyarı: Ana DataFrame'de {col} sütunu bulunamadı.")
        df_processed[col] = np.nan

# Modelleme için kullanılacak nihai sütunları belirle (var olan ve tamamen NaN olmayanlar)
df_model_pool = df_processed[all_relevant_columns].copy()
df_model_pool.dropna(axis=1, how='all', inplace=True)
final_columns_for_modeling = df_model_pool.columns.tolist()

if not final_columns_for_modeling or len(final_columns_for_modeling) < 2:
    print("Hata: Modelleme için yeterli sütun kalmadı. Script sonlandırılıyor.")
    exit()

print(f"Modelleme için kullanılacak sütunlar: {final_columns_for_modeling}")

# --- Senaryo 2: Tüm veriyi birleştir ve belirtilen hedefler için model eğit ---
print(f"\n--- SENARYO 2 BAŞLIYOR (Belirtilen hedefler için Random Forest) ---")

models = {
    "Decision Tree": DecisionTreeRegressor(random_state=42)
}
# Diğer modeller (Linear Regression, Gradient Boosting, Neural Network, XGBoost) istenirse buraya eklenebilir.

results_summary = {}

for target_col in targets_to_predict:
    if target_col not in final_columns_for_modeling:
        print(f"Uyarı: Hedef değişken '{target_col}' kullanılabilir sütunlar arasında değil. Atlanıyor.")
        continue

    print(f"\n--- Hedef Değişken: {target_col} ---")
    results_summary[target_col] = {}

    current_data_for_target = df_model_pool.copy() # Her hedef için temiz bir başlangıç

    # X (özellikler) ve y (hedef) ayır
    y_full = current_data_for_target[target_col].copy()
    # Özellikler, hedefin kendisi hariç `final_columns_for_modeling` içindeki tüm sütunlar olmalı
    feature_pool = [col for col in final_columns_for_modeling if col != target_col]
    if not feature_pool:
        print(f"    Uyarı: {target_col} için özellik sütunu kalmadı. Atlanıyor.")
        continue
    X_full = current_data_for_target[feature_pool].copy()
    
    # Hedef değişkende NaN olan satırları çıkar
    valid_target_indices = y_full.dropna().index
    y = y_full.loc[valid_target_indices]
    X = X_full.loc[valid_target_indices]

    if X.empty or y.empty or len(X) < 5: # Train/test split için yeterli satır kontrolü
        print(f"    Uyarı: {target_col} için hedefte NaN olmayan yeterli veri ({len(X)} satır) kalmadı. Atlanıyor.")
        continue
    
    # Veriyi eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if X_train.empty or X_test.empty:
        print(f"    Uyarı: {target_col} için train/test split sonrası boş küme(ler) oluştu. Atlanıyor.")
        continue
        
    # Eğitim setindeki (X_train) NaN değerleri medyan ile doldur
    feature_names = X_train.columns.tolist()
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_names)
    
    # Test setindeki (X_test) NaN değerleri eğitim setinden öğrenilen imputer ile doldur
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=feature_names)

    # İsteğe göre: Test bölümündeki (X_test_imputed ve y_test) NaN içeren satırları dropla
    y_test_reset = y_test.reset_index(drop=True)
    X_test_imputed_reset = X_test_imputed.reset_index(drop=True)
    
    y_test_name = y_test_reset.name if y_test_reset.name else 'target_variable'
    if y_test_name in X_test_imputed_reset.columns:
        y_test_name = f"{y_test_name}_y"

    temp_test_df = pd.concat([X_test_imputed_reset, y_test_reset.rename(y_test_name)], axis=1)
    cleaned_test_df = temp_test_df.dropna()

    if cleaned_test_df.empty:
        print(f"    Uyarı: {target_col} için test setinde NaN temizliği sonrası veri kalmadı. Atlanıyor.")
        continue
            
    X_test_final = cleaned_test_df.drop(columns=[y_test_name])
    y_test_final = cleaned_test_df[y_test_name]

    if X_test_final.empty or y_test_final.empty:
        print(f"    Uyarı: {target_col} için test seti (X veya y) temizlik sonrası boş kaldı. Atlanıyor.")
        continue
    if len(X_test_final) < 1:
         print(f"    Uyarı: {target_col} için test seti temizlik sonrası yetersiz ({len(X_test_final)} örnek). Atlanıyor.")
         continue
         
    # Özellikleri ölçeklendir (Linear Regression, Neural Network gibi modeller için önemli olabilir)
    # Random Forest için genellikle gerekmez ama tutarlılık için eklenebilir veya çıkarılabilir.
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train_imputed)
    # X_test_scaled = scaler.transform(X_test_final)
    # Kullanılacak X setleri:
    X_train_to_use = X_train_imputed # Ölçeklendirme yoksa
    X_test_to_use = X_test_final   # Ölçeklendirme yoksa
    # Eğer ölçeklendirme aktif edilirse:
    # X_train_to_use = X_train_scaled
    # X_test_to_use = X_test_scaled

    for model_name, model_instance in models.items():
        print(f"  Model Eğitiliyor: {model_name}...")
        try:
            current_model = model_instance # Modelin taze kopyası
            current_model.fit(X_train_to_use, y_train) # y_train NaN'lardan arındırılmıştı
            
            if X_test_to_use.shape[0] > 0:
                y_pred = current_model.predict(X_test_to_use)
                
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
                
                non_zero_mask = y_test_final != 0
                if np.any(non_zero_mask):
                    mape = mean_absolute_percentage_error(y_test_final[non_zero_mask], y_pred[non_zero_mask]) * 100
                    print("      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.")
                else:
                    print("      Bilgi: MAPE hesaplanamıyor (tüm y_test_final değerleri sıfır).")
                    mape = np.nan

                mape_str = f"{mape:.2f}%" if not np.isnan(mape) else "N/A"
                print(f"    {model_name}: MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}{r2_note}, MAPE={mape_str}")
                results_summary[target_col][model_name] = {
                    "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2 if not np.isnan(r2) else None, "MAPE": mape if not np.isnan(mape) else None
                }
                if r2_note:
                     results_summary[target_col][model_name]["Not_R2"] = r2_note.strip()
            else:                
                print(f"    {model_name}: Test verisi kalmadığı için metrikler hesaplanamadı.")
                results_summary[target_col][model_name] = {
                    "MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "R2": np.nan, "MAPE": np.nan,
                    "Not": "Yetersiz/Boş test verisi"
                }
        except Exception as e:
            print(f"    Hata ({model_name} modeli, {target_col} hedefi için): {e}")
            results_summary[target_col][model_name] = {"MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "R2": np.nan, "MAPE": np.nan, "Hata": str(e)}

# Sonuçların özetini yazdır (isteğe bağlı)
# ... (results_summary DataFrame'e dönüştürülüp yazdırılabilir) ...

print("\nSenaryo 2 tamamlandı.")