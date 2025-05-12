import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer # type: ignore
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

# Veri setini CSV dosyasından oku
# Dosya adını ve ayırıcıyı kendi veri setinize göre güncelleyin
# Eğer dosya script ile aynı dizinde değilse tam yolunu belirtin.
try:
    df = pd.read_csv('dataset_synop.csv', sep=';') # Ayırıcı noktalı virgül (;) olarak güncellendi.
except FileNotFoundError:
    print("Hata: 'dataset_synop.csv' dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
    exit()

print("--- Yüklenen DataFrame'in İlk 5 Satırı ---")
print(df.head())
print("\n--- Yüklenen DataFrame'in Sütun Adları ---")
print(df.columns.tolist()) # .tolist() ile daha okunaklı bir liste olarak yazdırır

# "X.AyAdı" formatındaki (örn: "1.May") değerleri sayısal değere çeviren fonksiyon
def parse_month_string_value(value):
    if isinstance(value, str):
        # Regex: başında sayı, sonra nokta, sonra bilinen ay kısaltmalarından biri
        match = re.match(r"(\d+)\.(?:Oca|Şub|Mar|Nis|May|Haz|Tem|Ağu|Eyl|Eki|Kas|Ara)", value, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return np.nan
    return np.nan # Format uyuşmuyorsa veya string değilse NaN döner

# Modellemede kullanılacak temel fiziksel ölçüm sütunları
# Sayısal olması beklenen ve görece temiz görünen sütunlar seçildi.
physical_measurement_columns = [
    'Sea level pressure',
    '3-hour pressure variation',
    '10-min mean wind speed', # Özel parse işlemi gerektiriyor
    'Temperature',           # Kelvin cinsinden sıcaklık
    'Dew point',             # Kelvin cinsinden çiğ noktası
    'Humidity',
    'Horizontal visibility',
    'Total cloud cover',
    'Station pressure',
    '24-hour pressure variation',
    'Precipitation in the last 24 hours' # Örnek bir yağış sütunu
]

# "X.AyAdı" formatında parse edilecek sütunlar
month_string_columns = ['10-min mean wind speed', 'Gust over last 10 minutes']

# --- Veri Ön İşleme ---
df_processed = df.copy()

# 1. "X.AyAdı" formatındaki sütunları özel parse fonksiyonu ile işle
for col in month_string_columns:
    if col in df_processed.columns:
        df_processed[col] = df_processed[col].apply(parse_month_string_value)

# 2. Seçilen tüm fiziksel ölçüm sütunlarını sayısal değere dönüştür, hataları NaN yap
for col in physical_measurement_columns:
    if col in df_processed.columns:
        # Eğer daha önce month_string_columns içinde işlenmediyse
        if col not in month_string_columns or pd.api.types.is_object_dtype(df_processed[col]):
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    else:
        print(f"Uyarı: {col} sütunu DataFrame'de bulunamadı.")
        # Gerekirse, anahtar hatası almamak için NaN içeren boş bir sütun eklenebilir
        # df_processed[col] = np.nan 

# Modelleme için sadece ilgili sütunları al
columns_for_modeling_present = [col for col in physical_measurement_columns if col in df_processed.columns]
df_model_data = df_processed[columns_for_modeling_present].copy()

# Tamamen NaN olan sütunları çıkar (eğer varsa)
df_model_data.dropna(axis=1, how='all', inplace=True)

# Eğer sütun çıkarıldıysa, modelleme sütun listesini güncelle
columns_for_modeling_final = [col for col in columns_for_modeling_present if col in df_model_data.columns]

if not columns_for_modeling_final:
    print("Uyarı: Modelleme için kullanılabilir hiçbir sayısal sütun bulunamadı.")
    exit()

# Eksik değerleri medyan ile doldur
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df_model_data), columns=df_model_data.columns)

# --- Senaryo 2: Tüm veriyi birleştir ve her fiziksel ölçüm için model eğit ---

# Makine Öğrenmesi Modelleri
models = {
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=10, max_depth=5) # Küçük veri için parametreler}
}

results_summary = {}

# Her bir fiziksel ölçüm sütununu hedef değişken olarak alıp modelleme yap
for target_col in columns_for_modeling_final:
    print(f"\n--- Hedef Değişken: {target_col} ---")
    results_summary[target_col] = {}

    # X (özellikler) ve y (hedef) değişkenlerini hazırla
    y = df_imputed[target_col]
    X = df_imputed.drop(columns=[target_col])

    if X.empty:
        print(f"Uyarı: {target_col} hedef değişkeni için özellik kalmadı. Atlanıyor.")
        continue
    
    # Veriyi eğitim ve test setlerine ayır
    # Örnek veri çok küçük olduğu için test_size'ı yüksek tutmak veya cross-validation yapmak daha iyi olabilir
    # Ancak şimdilik basit bir split yapıyoruz. Yeterli veri yoksa (örn: <2 sample testte) hata verebilir.
    if len(X) < 5 : # Eğer çok az satır varsa (özellikle split sonrası)
        print(f"Uyarı: {target_col} için yetersiz veri ({len(X)} satır). Train/test split atlanıyor, tüm veriyle eğitiliyor (önerilmez).")
        X_train, X_test, y_train, y_test = X, X, y, y # Geçici çözüm, normalde yapılmaz
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Özellikleri ölçeklendir (Lineer Regresyon ve Neural Network için önemli)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for model_name, model_instance in models.items():
        print(f"  Model Eğitiliyor: {model_name}...")
        try:
            current_model = model_instance # Her iterasyonda modelin taze bir kopyasını kullan
            if model_name in ["Linear Regression", "Neural Network (MLP)"]:
                current_model.fit(X_train_scaled, y_train)
                # Test verisi yoksa veya çok azsa tahmin yapma
                if X_test_scaled.shape[0] > 0:
                    y_pred = current_model.predict(X_test_scaled)
                else:
                    y_pred = [] # Tahmin yok
            else: # Ağaç tabanlı modeller ölçeklendirmeye daha az duyarlı
                current_model.fit(X_train, y_train)
                if X_test.shape[0] > 0:
                    y_pred = current_model.predict(X_test)
                else:
                    y_pred = []

            # Metrik hesaplama kısmı
            # X_test.shape[0] test setindeki örnek sayısını verir.
            # y_pred, X_test boşsa [] olarak ayarlanır.
            if X_test.shape[0] > 0: # Test setinde değerlendirilecek örnek varsa
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                mape = np.nan # Olası sıfıra bölme hatası için varsayılan
                try:
                    # y_test içinde 0 varsa MAPE tanımsız olabilir veya sklearn'in versiyonuna göre farklı davranabilir.
                    # sklearn.metrics.mean_absolute_percentage_error genellikle bunu iyi yönetir.
                    mape = mean_absolute_percentage_error(y_test, y_pred) * 100 # Yüzde olarak göstermek için
                except ZeroDivisionError:
                    print(f"    UYARI: {target_col} için MAPE hesaplanırken sıfıra bölme hatası (y_test'te 0 olabilir). MAPE NaN olarak ayarlandı.")
                
                r2_value = np.nan  # R2 için varsayılan değer NaN
                r2_note = ""

                if X_test.shape[0] >= 2: # R2 hesaplamak için en az 2 örnek gerekli
                    r2_value = r2_score(y_test, y_pred)
                    print(f"    {model_name}: MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2_value:.4f}, MAPE={mape:.2f}%")
                else: # Test setinde sadece 1 örnek var
                    r2_note = " (R² tanımsız: 1 örnek)"
                    print(f"    {model_name}: MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2_value}{r2_note}, MAPE={mape:.2f}%")
                
                results_summary[target_col][model_name] = {
                    "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2_value, "MAPE": mape
                }
                if r2_note: # Add note if R2 was undefined
                     results_summary[target_col][model_name]["Not_R2"] = r2_note.strip()
            else: # Test setinde hiç örnek yok (X_test.shape[0] == 0)                
                print(f"    {model_name}: Test verisi yetersiz olduğu için metrikler hesaplanamadı.")
                results_summary[target_col][model_name] = {
                    "MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "R2": np.nan, "MAPE": np.nan,
                    "Not": "Yetersiz test verisi"
                }

        except Exception as e:
            print(f"    Hata ({model_name} modeli, {target_col} hedefi için): {e}")
            results_summary[target_col][model_name] = {"MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "R2": np.nan, "MAPE": np.nan, "Hata": str(e)}

# Sonuçların özetini yazdır
print("\n--- Sonuçların Özeti (MAE) ---")
summary_mae_df = pd.DataFrame({target: {model_name: res.get("MAE", np.nan)
                                      for model_name, res in models_res.items()}
                            for target, models_res in results_summary.items()}).T
print(summary_mae_df)

print("\n--- Sonuçların Özeti (MSE) ---")
summary_mse_df = pd.DataFrame({target: {model_name: res.get("MSE", np.nan)
                                      for model_name, res in models_res.items()}
                            for target, models_res in results_summary.items()}).T
print(summary_mse_df)

print("\n--- Sonuçların Özeti (RMSE) ---")
summary_rmse_df = pd.DataFrame({target: {model_name: res.get("RMSE", np.nan)
                                      for model_name, res in models_res.items()}
                            for target, models_res in results_summary.items()}).T
print(summary_rmse_df)

print("\n--- Sonuçların Özeti (R2 Skoru) ---")
summary_r2_df = pd.DataFrame({target: {model_name: res.get("R2", np.nan) 
                                     for model_name, res in models_res.items()}
                           for target, models_res in results_summary.items()}).T
print(summary_r2_df)

print("\n--- Sonuçların Özeti (MAPE %) ---")
summary_mape_df = pd.DataFrame({target: {model_name: res.get("MAPE", np.nan)
                                      for model_name, res in models_res.items()}
                            for target, models_res in results_summary.items()}).T
print(summary_mape_df)

print("\nScript tamamlandı.")