# Federated Learning on Weather Station Data

# RANDOM FOREST

## 9 Mayıs Öncesi 

### SENARYO 1

#### CLEAN

İstasyon 7761 için sonuçlar:
Mean Absolute Error (MAE): 3.686579937691042
Mean Squared Error (MSE): 21.800469797185357
Root Mean Squared Error (RMSE): 4.669097321451477
R^2 Score: 0.5329069732890008

İstasyon 7790 için sonuçlar:
Mean Absolute Error (MAE): 3.9943225258260386
Mean Squared Error (MSE): 25.189406406664958
Root Mean Squared Error (RMSE): 5.018904901137793
R^2 Score: 0.46460385054868425

#### MEAN

İstasyon 7761 için sonuçlar:
Mean Absolute Error (MAE): 0.00023078677423129127
Mean Squared Error (MSE): 3.798271615650748e-05
Root Mean Squared Error (RMSE): 0.006163011938695842
R^2 Score: 0.9999991960571203

İstasyon 7790 için sonuçlar:
Mean Absolute Error (MAE): 0.00011286705021471535
Mean Squared Error (MSE): 9.735229470728015e-06
Root Mean Squared Error (RMSE): 0.003120132925169698
R^2 Score: 0.9999997885622143

#### MEDIAN

İstasyon 7761 için sonuçlar:
Mean Absolute Error (MAE): 0.00022040864257537237
Mean Squared Error (MSE): 3.205342883983392e-05
Root Mean Squared Error (RMSE): 0.005661574766779462
R^2 Score: 0.9999993215578054

İstasyon 7790 için sonuçlar:
Mean Absolute Error (MAE): 0.00011023622049244808
Mean Squared Error (MSE): 9.1185803267159e-06
Root Mean Squared Error (RMSE): 0.0030196987145600968
R^2 Score: 0.9999998019588758

#### Interpolasyon ve Hiperparametre ayarı yapılmış Hal

İstasyon 7761 için sonuçlar:
Mean Absolute Error (MAE): 0.0020683311467548427
Mean Squared Error (MSE): 0.0035206425321130472
Root Mean Squared Error (RMSE): 0.05933500258795855
R^2 Score: 0.9999255076370822

İstasyon 7790 için sonuçlar:
Mean Absolute Error (MAE): 0.021427187980687722
Mean Squared Error (MSE): 0.09140171125580605
Root Mean Squared Error (RMSE): 0.30232715930892823
R^2 Score: 0.9980234195781912

R^2 skorları çok iyi, overfitting ihtimali??

Untitled-1
Random Forest - MAE: 0.0012570847171043465, R²: 0.999501490611814

### SENARYO 1

--- İstasyon ID: 7761 ---
  --- Hedef Değişken: Sea level pressure ---
    Random Forest: MAE=0.6801, MSE=132.4676, RMSE=11.5095, R2=0.9997, MAPE=0.00%
  --- Hedef Değişken: 3-hour pressure variation ---
    Random Forest: MAE=69.1767, MSE=8300.7089, RMSE=91.1082, R2=0.1755, MAPE=580084362524976256.00%
  --- Hedef Değişken: Temperature ---
    Random Forest: MAE=0.0369, MSE=0.0126, RMSE=0.1123, R2=0.9997, MAPE=0.01%
  --- Hedef Değişken: Dew point ---
    Random Forest: MAE=0.0349, MSE=0.0138, RMSE=0.1176, R2=0.9996, MAPE=0.01%
  --- Hedef Değişken: Humidity ---
    Random Forest: MAE=0.1728, MSE=0.1082, RMSE=0.3290, R2=0.9994, MAPE=0.27%
  --- Hedef Değişken: Horizontal visibility ---
    Random Forest: MAE=11072.6237, MSE=185294458.1328, RMSE=13612.2907, R2=0.2485, MAPE=55.69%
  --- Hedef Değişken: Total cloud cover ---
    Random Forest: MAE=13.2327, MSE=453.0258, RMSE=21.2844, R2=0.5219, MAPE=859158935441757696.00%
  --- Hedef Değişken: Station pressure ---
    Random Forest: MAE=0.6181, MSE=177.3201, RMSE=13.3162, R2=0.9996, MAPE=0.00%
  --- Hedef Değişken: 24-hour pressure variation ---
    Random Forest: MAE=199.3723, MSE=89118.9334, RMSE=298.5279, R2=0.3064, MAPE=11094970296420982784.00%
  --- Hedef Değişken: Precipitation in the last 24 hours ---
    Random Forest: MAE=1.3578, MSE=14.0982, RMSE=3.7548, R2=0.2907, MAPE=181757210865521440.00%

--- İstasyon ID: 7790 ---
  --- Hedef Değişken: Sea level pressure ---
    Random Forest: MAE=4.4050, MSE=1516.2977, RMSE=38.9397, R2=0.9972, MAPE=0.00%
  --- Hedef Değişken: 3-hour pressure variation ---
    Random Forest: MAE=68.7032, MSE=8197.1513, RMSE=90.5381, R2=0.2041, MAPE=590088520925533696.00%
  --- Hedef Değişken: Temperature ---
    Random Forest: MAE=0.0446, MSE=0.0216, RMSE=0.1469, R2=0.9995, MAPE=0.02%
  --- Hedef Değişken: Dew point ---
    Random Forest: MAE=0.0385, MSE=0.0153, RMSE=0.1235, R2=0.9996, MAPE=0.01%
  --- Hedef Değişken: Humidity ---
    Random Forest: MAE=0.2987, MSE=0.3992, RMSE=0.6318, R2=0.9984, MAPE=0.47%
  --- Hedef Değişken: Horizontal visibility ---
    Random Forest: MAE=11324.2931, MSE=195098639.4907, RMSE=13967.7715, R2=0.2976, MAPE=1285693274277037568.00%
  --- Hedef Değişken: Total cloud cover ---
    Random Forest: MAE=15.6849, MSE=498.0152, RMSE=22.3163, R2=0.4073, MAPE=937737407897214976.00%
  --- Hedef Değişken: Station pressure ---
    Random Forest: MAE=3.6872, MSE=1728.1217, RMSE=41.5707, R2=0.9968, MAPE=0.00%
  --- Hedef Değişken: 24-hour pressure variation ---
    Random Forest: MAE=215.2953, MSE=100895.0079, RMSE=317.6397, R2=0.3502, MAPE=15701017567973898240.00%
  --- Hedef Değişken: Precipitation in the last 24 hours ---
    Random Forest: MAE=1.6165, MSE=34.9459, RMSE=5.9115, R2=0.0647, MAPE=257944475801778752.00%

### SENARYO 2

--- Sonuçların Özeti (MAE) ---
                                    Random Forest
Sea level pressure                      28.290346
3-hour pressure variation               68.854730
Temperature                              0.987237
Dew point                                0.896784
Humidity                                 6.112490
Horizontal visibility                11809.904163
Total cloud cover                       16.945370
Station pressure                        25.063366
24-hour pressure variation             223.801681
Precipitation in the last 24 hours       1.533372

--- Sonuçların Özeti (MSE) ---
                                    Random Forest
Sea level pressure                   3.331343e+03
3-hour pressure variation            8.169998e+03
Temperature                          1.837412e+00
Dew point                            1.417500e+00
Humidity                             6.341180e+01
Horizontal visibility                2.032556e+08
Total cloud cover                    5.836371e+02
Station pressure                     3.113472e+03
24-hour pressure variation           1.078617e+05
Precipitation in the last 24 hours   2.278765e+01

--- Sonuçların Özeti (RMSE) ---
                                    Random Forest
Sea level pressure                      57.717788
3-hour pressure variation               90.388041
Temperature                              1.355512
Dew point                                1.190588
Humidity                                 7.963152
Horizontal visibility                14256.774611
Total cloud cover                       24.158583
Station pressure                        55.798494
24-hour pressure variation             328.423086
Precipitation in the last 24 hours       4.773641

--- Sonuçların Özeti (R2 Skoru) ---
                                    Random Forest
Sea level pressure                       0.993390
3-hour pressure variation                0.172773
Temperature                              0.960515
Dew point                                0.962142
Humidity                                 0.720138
Horizontal visibility                    0.238337
Total cloud cover                        0.280217
Station pressure                         0.993839
24-hour pressure variation               0.238600
Precipitation in the last 24 hours       0.111891

--- Sonuçların Özeti (MAPE %) ---
                                    Random Forest
Sea level pressure                   2.786377e-02
3-hour pressure variation            4.451907e+17
Temperature                          3.415008e-01
Dew point                            3.161096e-01
Humidity                             9.096868e+00
Horizontal visibility                5.881757e+01
Total cloud cover                    1.162723e+18
Station pressure                     2.472655e-02
24-hour pressure variation           1.509705e+19
Precipitation in the last 24 hours   2.394178e+17

## 9 MAYIS 2025 TARİHLİ TOPLANTI

  rüzgar hızı, nem, sıcaklık, rüzgar yönü
  hedef nan drop
  parametre nan median

## 12 MAYIS 2025

### SENARYO 1

--- SENARYO 1 BAŞLIYOR (Belirtilen hedefler için Random Forest) ---
Toplam 2 istasyon bulundu.

--- İstasyon ID: 7761 ---
  --- Hedef Değişken: 10-min mean wind speed ---
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Random Forest: MAE=0.8465, MSE=1.4355, RMSE=1.1981, R2=0.4574, MAPE=38.05%
  --- Hedef Değişken: Humidity ---
    Random Forest: MAE=0.4136, MSE=0.8138, RMSE=0.9021, R2=0.9958, MAPE=0.62%
  --- Hedef Değişken: Temperature ---
    Random Forest: MAE=0.0375, MSE=0.0097, RMSE=0.0987, R2=0.9998, MAPE=0.01%
  --- Hedef Değişken: 10-min mean wind direction ---
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Random Forest: MAE=39.3305, MSE=3686.2306, RMSE=60.7143, R2=0.5946, MAPE=62.00%

--- İstasyon ID: 7790 ---
  --- Hedef Değişken: 10-min mean wind speed ---
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Random Forest: MAE=0.9214, MSE=1.7606, RMSE=1.3269, R2=0.4671, MAPE=46.36%
  --- Hedef Değişken: Humidity ---
    Random Forest: MAE=0.6264, MSE=1.9806, RMSE=1.4073, R2=0.9924, MAPE=0.94%
  --- Hedef Değişken: Temperature ---
    Random Forest: MAE=0.0458, MSE=0.0243, RMSE=0.1558, R2=0.9995, MAPE=0.02%
  --- Hedef Değişken: 10-min mean wind direction ---
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Random Forest: MAE=50.6864, MSE=5432.6715, RMSE=73.7067, R2=0.3130, MAPE=86.00%

Senaryo 1 tamamlandı.

### SENARYO 2

--- SENARYO 2 BAŞLIYOR (Belirtilen hedefler için Random Forest) ---

--- Hedef Değişken: 10-min mean wind speed ---
  Model Eğitiliyor: Random Forest...
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Random Forest: MAE=0.9289, MSE=1.7565, RMSE=1.3253, R2=0.4244, MAPE=45.08%

--- Hedef Değişken: Humidity ---
  Model Eğitiliyor: Random Forest...
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Random Forest: MAE=0.2641, MSE=0.5183, RMSE=0.7199, R2=0.9977, MAPE=0.38%

--- Hedef Değişken: Temperature ---
  Model Eğitiliyor: Random Forest...
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Random Forest: MAE=0.0242, MSE=0.0082, RMSE=0.0908, R2=0.9998, MAPE=0.01%

--- Hedef Değişken: 10-min mean wind direction ---
  Model Eğitiliyor: Random Forest...
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Random Forest: MAE=66.4049, MSE=6876.9718, RMSE=82.9275, R2=0.2694, MAPE=106.46%

Senaryo 2 tamamlandı.

### TAVSİYELER

Rüzgar yönü tahmini için:
  Açısal veriye özel yöntemler (örn. sine-cosine dönüşümü) denenebilir.
  Alternatif modeller (örn. Circular Regression, SVM) test edilebilir.

Rüzgar hızı için:
  Özellik mühendisliği geliştirilebilir (örn. rüzgar yönü ve basınç etkileşimi).
  Model karşılaştırması yapıldığında belki XGBoost veya Gradient Boosting daha iyi olabilir.

Model karşılaştırması:
  Kodda sadece Random Forest denenmiş. Diğer algoritmalar da test edilerek karşılaştırma yapılmalı (bu zaten senaryonun devamı olacak gibi duruyor).

MAPE değerleri yüksek olanlar için:
  Verinin normalize edilmesi veya outlier temizliği fayda sağlayabilir.

### RÜZGAR YÖNÜ VE HIZI TAHMİNLERİ İÇİN VERİLEN TAVSİYELER UYGULANINCA

#### SENARYO 1

--- SENARYO 1 BAŞLIYOR (Belirtilen hedefler için Random Forest) ---
Toplam 2 istasyon bulundu.

--- İstasyon ID: 7761 ---
  --- Hedef Değişken: 10-min mean wind speed ---
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Random Forest: MAE=0.8491, MSE=1.4515, RMSE=1.2048, R2=0.4514, MAPE=38.08%
  --- Hedef Değişken: Humidity ---
    Random Forest: MAE=0.4253, MSE=0.8488, RMSE=0.9213, R2=0.9956, MAPE=0.63%
  --- Hedef Değişken: Temperature ---
    Random Forest: MAE=0.0399, MSE=0.0117, RMSE=0.1081, R2=0.9998, MAPE=0.01%
  --- Hedef Değişken: 10-min mean wind direction ---
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Random Forest: MAE=0.0013, MSE=0.0044, RMSE=0.0663, R2=1.0000, MAPE=0.00%

--- İstasyon ID: 7790 ---
  --- Hedef Değişken: 10-min mean wind speed ---
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Random Forest: MAE=0.9241, MSE=1.7696, RMSE=1.3303, R2=0.4644, MAPE=46.47%
  --- Hedef Değişken: Humidity ---
    Random Forest: MAE=0.7039, MSE=2.1878, RMSE=1.4791, R2=0.9916, MAPE=1.08%
  --- Hedef Değişken: Temperature ---
    Random Forest: MAE=0.0488, MSE=0.0395, RMSE=0.1988, R2=0.9991, MAPE=0.02%
  --- Hedef Değişken: 10-min mean wind direction ---
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Random Forest: MAE=0.0004, MSE=0.0004, RMSE=0.0190, R2=1.0000, MAPE=0.00%

Senaryo 1 tamamlandı.

#### SENARYO 2

--- SENARYO 2 BAŞLIYOR (Belirtilen hedefler için Random Forest) ---

--- Hedef Değişken: 10-min mean wind speed ---
  Model Eğitiliyor: Random Forest...
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Random Forest: MAE=0.9271, MSE=1.7488, RMSE=1.3224, R2=0.4269, MAPE=44.89%

--- Hedef Değişken: Humidity ---
  Model Eğitiliyor: Random Forest...
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Random Forest: MAE=0.2861, MSE=0.6635, RMSE=0.8145, R2=0.9971, MAPE=0.41%

--- Hedef Değişken: Temperature ---
  Model Eğitiliyor: Random Forest...
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Random Forest: MAE=0.0248, MSE=0.0081, RMSE=0.0902, R2=0.9998, MAPE=0.01%

--- Hedef Değişken: 10-min mean wind direction ---
  Model Eğitiliyor: Random Forest...
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Random Forest: MAE=0.0006, MSE=0.0014, RMSE=0.0373, R2=1.0000, MAPE=0.00%

Senaryo 2 tamamlandı.

SENARYO 1 VE 2'DE RÜZGAR YÖNÜ İÇİN HARİKA SONUÇLAR ALINDI ANCAK RÜZGAR HIZINDA İŞLER HALA İYİ DEĞİL

### RÜZGAR HIZI İÇİN wind_dir_cos × horizontal visibility KOMBİNASYONU DENENİYOR

#### SENARYO 2

--- SENARYO 2 BAŞLIYOR (Belirtilen hedefler için Random Forest) ---

--- Hedef Değişken: 10-min mean wind speed ---
  Model Eğitiliyor: Random Forest...
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Random Forest: MAE=0.9287, MSE=1.7521, RMSE=1.3237, R2=0.4258, MAPE=44.92%

--- Hedef Değişken: Humidity ---
  Model Eğitiliyor: Random Forest...
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Random Forest: MAE=0.2849, MSE=0.6389, RMSE=0.7993, R2=0.9972, MAPE=0.41%

--- Hedef Değişken: Temperature ---
  Model Eğitiliyor: Random Forest...
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Random Forest: MAE=0.0247, MSE=0.0080, RMSE=0.0897, R2=0.9998, MAPE=0.01%

--- Hedef Değişken: 10-min mean wind direction ---
  Model Eğitiliyor: Random Forest...
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Random Forest: MAE=0.0003, MSE=0.0005, RMSE=0.0223, R2=1.0000, MAPE=0.00%

Senaryo 2 tamamlandı.

Hiçbir değişim yok, değişikliklerden vazgeçildi

### DECISION TREE

#### SENARYO 1

--- SENARYO 1 BAŞLIYOR (Belirtilen hedefler için Random Forest) ---
Toplam 2 istasyon bulundu.

--- İstasyon ID: 7761 ---
  --- Hedef Değişken: 10-min mean wind speed ---
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Random Forest: MAE=1.1960, MSE=2.9237, RMSE=1.7099, R2=-0.1051, MAPE=49.25%
  --- Hedef Değişken: Humidity ---
    Random Forest: MAE=0.7008, MSE=2.3734, RMSE=1.5406, R2=0.9877, MAPE=1.05%
  --- Hedef Değişken: Temperature ---
    Random Forest: MAE=0.0706, MSE=0.0546, RMSE=0.2337, R2=0.9988, MAPE=0.02%
  --- Hedef Değişken: 10-min mean wind direction ---
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Random Forest: MAE=0.0000, MSE=0.0000, RMSE=0.0000, R2=1.0000, MAPE=0.00%

--- İstasyon ID: 7790 ---
  --- Hedef Değişken: 10-min mean wind speed ---
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Random Forest: MAE=1.2804, MSE=3.5044, RMSE=1.8720, R2=-0.0606, MAPE=59.30%
  --- Hedef Değişken: Humidity ---
    Random Forest: MAE=1.3643, MSE=6.6267, RMSE=2.5742, R2=0.9745, MAPE=2.08%
  --- Hedef Değişken: Temperature ---
    Random Forest: MAE=0.0890, MSE=0.0989, RMSE=0.3145, R2=0.9979, MAPE=0.03%
  --- Hedef Değişken: 10-min mean wind direction ---
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Random Forest: MAE=0.0000, MSE=0.0000, RMSE=0.0000, R2=1.0000, MAPE=0.00%

Senaryo 1 tamamlandı.

#### SENARYO 2

--- SENARYO 2 BAŞLIYOR (Belirtilen hedefler için Random Forest) ---

--- Hedef Değişken: 10-min mean wind speed ---
  Model Eğitiliyor: Decision Tree...
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Decision Tree: MAE=1.3170, MSE=3.6520, RMSE=1.9110, R2=-0.1968, MAPE=58.49%

--- Hedef Değişken: Humidity ---
  Model Eğitiliyor: Decision Tree...
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Decision Tree: MAE=0.4488, MSE=1.8933, RMSE=1.3760, R2=0.9917, MAPE=0.64%

--- Hedef Değişken: Temperature ---
  Model Eğitiliyor: Decision Tree...
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Decision Tree: MAE=0.0418, MSE=0.0411, RMSE=0.2028, R2=0.9991, MAPE=0.01%

--- Hedef Değişken: 10-min mean wind direction ---
  Model Eğitiliyor: Decision Tree...
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Decision Tree: MAE=0.0000, MSE=0.0000, RMSE=0.0000, R2=1.0000, MAPE=0.00%

Senaryo 2 tamamlandı.

### XGBOOST

#### SENARYO 1

--- SENARYO 1 BAŞLIYOR (Belirtilen hedefler için Random Forest) ---
Toplam 2 istasyon bulundu.

--- İstasyon ID: 7761 ---
  --- Hedef Değişken: 10-min mean wind speed ---
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Random Forest: MAE=0.8478, MSE=1.4531, RMSE=1.2054, R2=0.4508, MAPE=38.06%
  --- Hedef Değişken: Humidity ---
    Random Forest: MAE=0.8064, MSE=1.3630, RMSE=1.1675, R2=0.9930, MAPE=1.21%
  --- Hedef Değişken: Temperature ---
    Random Forest: MAE=0.0922, MSE=0.0304, RMSE=0.1743, R2=0.9994, MAPE=0.03%
  --- Hedef Değişken: 10-min mean wind direction ---
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Random Forest: MAE=0.0024, MSE=0.0000, RMSE=0.0026, R2=1.0000, MAPE=0.00%

--- İstasyon ID: 7790 ---
  --- Hedef Değişken: 10-min mean wind speed ---
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Random Forest: MAE=0.9260, MSE=1.7392, RMSE=1.3188, R2=0.4736, MAPE=46.24%
  --- Hedef Değişken: Humidity ---
    Random Forest: MAE=1.0812, MSE=2.1432, RMSE=1.4640, R2=0.9918, MAPE=1.65%
  --- Hedef Değişken: Temperature ---
    Random Forest: MAE=0.0937, MSE=0.0484, RMSE=0.2200, R2=0.9990, MAPE=0.03%
  --- Hedef Değişken: 10-min mean wind direction ---
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Random Forest: MAE=0.0020, MSE=0.0000, RMSE=0.0024, R2=1.0000, MAPE=0.00%

Senaryo 1 tamamlandı.

#### SENARYO 2

--- SENARYO 2 BAŞLIYOR (Belirtilen hedefler için Random Forest) ---

--- Hedef Değişken: 10-min mean wind speed ---
  Model Eğitiliyor: XGBoost...
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    XGBoost: MAE=0.9421, MSE=1.7837, RMSE=1.3356, R2=0.4154, MAPE=45.28%

--- Hedef Değişken: Humidity ---
  Model Eğitiliyor: XGBoost...
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    XGBoost: MAE=0.5002, MSE=0.5642, RMSE=0.7512, R2=0.9975, MAPE=0.77%

--- Hedef Değişken: Temperature ---
  Model Eğitiliyor: XGBoost...
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    XGBoost: MAE=0.0831, MSE=0.0226, RMSE=0.1505, R2=0.9995, MAPE=0.03%

--- Hedef Değişken: 10-min mean wind direction ---
  Model Eğitiliyor: XGBoost...
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    XGBoost: MAE=0.0023, MSE=0.0000, RMSE=0.0026, R2=1.0000, MAPE=0.00%

Senaryo 2 tamamlandı.

Hala kötü sonuç var rüzgar hızı için, bunun için iyileştirmeler denenecek.

#### İYİLEŞTİRMELER DENENDİ

--- SENARYO 1 BAŞLIYOR (Belirtilen hedefler için XGBoost) ---
Toplam 2 istasyon bulundu.

--- İstasyon ID: 7761 ---
  --- Hedef Değişken: 10-min mean wind speed ---
      Bilgi: MAPE sıfır olmayan değerler için hesaplandı.
    XGBoost: MAE=0.8595, MSE=1.5291, RMSE=1.2366, R2=0.4220, MAPE=35.88%
  --- Hedef Değişken: Humidity ---
    XGBoost: MAE=1.3591, MSE=3.3648, RMSE=1.8344, R2=0.9826, MAPE=1.93%
  --- Hedef Değişken: Temperature ---
    XGBoost: MAE=0.1910, MSE=0.0732, RMSE=0.2706, R2=0.9984, MAPE=0.07%
  --- Hedef Değişken: 10-min mean wind direction ---
      Bilgi: MAPE sıfır olmayan değerler için hesaplandı.
    XGBoost: MAE=0.1492, MSE=0.2894, RMSE=0.5379, R2=1.0000, MAPE=0.10%

--- İstasyon ID: 7790 ---
  --- Hedef Değişken: 10-min mean wind speed ---
      Bilgi: MAPE sıfır olmayan değerler için hesaplandı.
    XGBoost: MAE=0.9253, MSE=1.8483, RMSE=1.3595, R2=0.4406, MAPE=42.63%
  --- Hedef Değişken: Humidity ---
    XGBoost: MAE=1.5594, MSE=4.0617, RMSE=2.0154, R2=0.9844, MAPE=2.25%
  --- Hedef Değişken: Temperature ---
    XGBoost: MAE=0.2218, MSE=0.1065, RMSE=0.3264, R2=0.9977, MAPE=0.08%
  --- Hedef Değişken: 10-min mean wind direction ---
      Bilgi: MAPE sıfır olmayan değerler için hesaplandı.
    XGBoost: MAE=0.2111, MSE=0.2053, RMSE=0.4531, R2=1.0000, MAPE=0.13%

Senaryo 1 tamamlandı.

Neredeyse hiçbir değişim yok, değişikliklerden vazgeçildi.

## 13 MAYIS 2025

### RANDOM FOREST

#### VALIDATION DATA İLE EĞİTİM

##### SENARYO 1

--- SENARYO 1 BAŞLIYOR (Belirtilen hedefler için Random Forest) ---
Toplam 2 istasyon bulundu.

--- İstasyon ID: 7761 ---
  --- Hedef Değişken: 10-min mean wind speed ---
    [VALIDATION] MAE=0.8718, MSE=1.4967, RMSE=1.2234, R2=0.44295107929101385
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Random Forest: MAE=0.8554, MSE=1.4650, RMSE=1.2104, R2=0.4636, MAPE=38.37%
  --- Hedef Değişken: Humidity ---
    [VALIDATION] MAE=0.5156, MSE=1.3268, RMSE=1.1519, R2=0.9932815171804111
    Random Forest: MAE=0.4987, MSE=1.1311, RMSE=1.0635, R2=0.9941, MAPE=0.76%
  --- Hedef Değişken: Temperature ---
    [VALIDATION] MAE=0.0510, MSE=0.0276, RMSE=0.1661, R2=0.9994123892358171
    Random Forest: MAE=0.0510, MSE=0.0279, RMSE=0.1671, R2=0.9994, MAPE=0.02%
  --- Hedef Değişken: 10-min mean wind direction ---
    [VALIDATION] MAE=0.0002, MSE=0.0002, RMSE=0.0157, R2=0.9999999724971429
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Random Forest: MAE=0.0057, MSE=0.0666, RMSE=0.2580, R2=1.0000, MAPE=0.00%

--- İstasyon ID: 7790 ---
  --- Hedef Değişken: 10-min mean wind speed ---
    [VALIDATION] MAE=0.9213, MSE=1.7240, RMSE=1.3130, R2=0.45247448132473334
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Random Forest: MAE=0.9196, MSE=1.7710, RMSE=1.3308, R2=0.4736, MAPE=46.40%
  --- Hedef Değişken: Humidity ---
    [VALIDATION] MAE=0.7874, MSE=2.3477, RMSE=1.5322, R2=0.9908280697958882
    Random Forest: MAE=0.7870, MSE=2.4385, RMSE=1.5616, R2=0.9908, MAPE=1.18%
  --- Hedef Değişken: Temperature ---
    [VALIDATION] MAE=0.0541, MSE=0.0294, RMSE=0.1715, R2=0.9993748329428025
    Random Forest: MAE=0.0542, MSE=0.0169, RMSE=0.1301, R2=0.9996, MAPE=0.02%
  --- Hedef Değişken: 10-min mean wind direction ---
    [VALIDATION] MAE=0.0015, MSE=0.0055, RMSE=0.0739, R2=0.9999992993620446
      Bilgi: MAPE, y_test_final'daki sıfır olmayan değerler için hesaplandı.
    Random Forest: MAE=0.0022, MSE=0.0060, RMSE=0.0772, R2=1.0000, MAPE=0.00%

Senaryo 1 tamamlandı.

##### SENARYO 2

--- SENARYO 2 BAŞLIYOR (Validation ile Random Forest) ---

--- Hedef Değişken: 10-min mean wind speed ---
  Model eğitiliyor: Random Forest
    Val: MAE=0.9448, RMSE=1.3390, R2=0.42556060983995103
    Test: MAE=0.9368, RMSE=1.3278, R2=0.41284090493242587, MAPE=45.98%

--- Hedef Değişken: Humidity ---
  Model eğitiliyor: Random Forest
    Val: MAE=0.3244, RMSE=0.8121, R2=0.9971111602940029
    Test: MAE=0.3381, RMSE=0.9236, R2=0.9962163164574974, MAPE=0.48%

--- Hedef Değişken: Temperature ---
  Model eğitiliyor: Random Forest
    Val: MAE=0.0334, RMSE=0.1374, R2=0.999590676185685
    Test: MAE=0.0330, RMSE=0.1476, R2=0.9995305791614, MAPE=0.01%

--- Hedef Değişken: 10-min mean wind direction ---
  Model eğitiliyor: Random Forest
    Val: MAE=0.0004, RMSE=0.0183, R2=0.9999999646025075
    Test: MAE=0.0002, RMSE=0.0236, R2=0.9999999416701866, MAPE=0.00%

Senaryo 2 tamamlandı.