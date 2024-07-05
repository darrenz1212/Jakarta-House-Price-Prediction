# -*- coding: utf-8 -*-
"""TubesML

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13XzbLrthbwCyc891-PZESAI70o0gq2c_

## **Data Import**
"""

import pandas as pd

filepath = 'data_rumah2.tsv'
df = pd.read_csv(filepath, sep='\t')
df

print("Harga rumah tertinggi", max(df['harga rumah']))
print("Harga rumah terrendah",min(df['harga rumah']))

"""## **Data Preprocessing**"""

df = df.drop(columns=['No','NRP','Nama','Link Data Rumah'])

df

print("Harga rumah tertinggi", max(df['harga rumah']))
print("Harga rumah terrendah",min(df['harga rumah']))

from sklearn.preprocessing import LabelEncoder
kolom_kategori = ['kecamatan','kelurahan','keamanan (ada/tidak)', 'taman (ada/tidak)','Kabupaten/Kota']
label_encoders = {}
for kolom in kolom_kategori:
    le = LabelEncoder()
    df[kolom] = le.fit_transform(df[kolom])
    label_encoders[kolom] = le

kolom_jarak = ['pasokan listrik (watt)','jumlah kamar tidur','jumlah kamar mandi','luas tanah (m2)','carport (mobil)','jarak dengan rumah sakit terdekat (km)', 'jarak dengan sekolah terdekat (km)', 'jarak dengan tol terdekat (km)']
for kolom in kolom_jarak:
    df[kolom] = df[kolom].astype(str)
    df[kolom] = df[kolom].str.replace(',', '.')
    df[kolom] = df[kolom].str.replace('-', ' ')
    df[kolom] = pd.to_numeric(df[kolom], errors='coerce')
    df = df.dropna()

df

df['taman (ada/tidak)']

df['harga rumah'] = df['harga rumah'].astype(str)
df['harga rumah'] = df['harga rumah'].str.replace('[^0-9]', '', regex=True)
df['harga rumah'] = df['harga rumah'].astype(int)

df['jarak dengan rumah sakit terdekat (km)'] = df['jarak dengan rumah sakit terdekat (km)'].astype(float)
df['jarak dengan sekolah terdekat (km)'] = df['jarak dengan sekolah terdekat (km)'].astype(float)
df['jarak dengan tol terdekat (km)'] = df['jarak dengan tol terdekat (km)'].astype(float)
df['jumlah kamar tidur'] = df['jumlah kamar tidur'].astype(float)
df['jumlah kamar mandi'] = df['jumlah kamar mandi'].astype(float)
df['luas tanah (m2)'] = df['luas tanah (m2)'].astype(float)
df['carport (mobil)'] = df['carport (mobil)'].astype(float)
df['pasokan listrik (watt)'] = df['pasokan listrik (watt)'].astype(float)
# df['harga rumah'] = df['harga rumah'].astype(int)

import numpy as np
df['harga rumah']

print("Harga rumah tertinggi", max(df['harga rumah']))
print("Harga rumah terrendah",min(df['harga rumah']))

df['harga rumah log'] = np.log1p(df['harga rumah'])

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 2.0 * IQR)) |(df > (Q3 + 2.0 * IQR))).any(axis=1)]

print("Harga rumah tertinggi", max(df['harga rumah']))
print("Harga rumah terrendah",min(df['harga rumah']))

# from scipy.stats import zscore
# df['zscore_harga_rumah_log'] = zscore(df['harga rumah log'])
# df = df[(df['zscore_harga_rumah_log'] <= 3) & (df['zscore_harga_rumah_log'] >= -3)]
# df = df.drop(columns=['zscore_harga_rumah_log'])

df

X = df.drop(['harga rumah', 'harga rumah log'], axis=1)
y = df['harga rumah log']

df.dtypes

"""all data is convert into float/integer"""

y

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.decomposition import PCA

principal=PCA()
X_scaled=principal.fit_transform(X_scaled)

X_scaled.shape

"""PCA Varians plot"""

import matplotlib.pyplot as plt
explained_variance = np.cumsum(principal.explained_variance_ratio_)
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance * 100, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.title('Cumulative Explained Variance by PCA Components')
plt.grid(True)
plt.show()

"""Dari hasil plot varians yang dijelaskan oleh komponen PCA, terlihat bahwa menggunakan 10 komponen utama sudah dapat menjelaskan hampir 90% dari total varians dalam data.

Applying PCA
"""

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

from sklearn.decomposition import PCA

principal=PCA(n_components=10)
X=principal.fit_transform(X)

X.shape

from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

import matplotlib.pyplot as plt

train_pca_df = pd.DataFrame(X_train, columns=[f'PC{i+1}' for i in range(10)])
train_pca_df['harga rumah'] = y_train
test_pca_df = pd.DataFrame(X_test, columns=[f'PC{i+1}' for i in range(10)])
test_pca_df['harga rumah'] = y_test

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
scatter_train = plt.scatter(train_pca_df['PC1'], train_pca_df['PC2'], c=train_pca_df['harga rumah'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter_train, label='Harga Rumah Log')
plt.title('PCA Result for Training Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.subplot(1, 2, 2)
scatter_test = plt.scatter(test_pca_df['PC1'], test_pca_df['PC2'], c=test_pca_df['harga rumah'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter_test, label='Harga Rumah Log')
plt.title('PCA Result for Testing Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.tight_layout()
plt.show()

"""## **Linear Reggression**"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

y_pred_lr = model_lr.predict(X_test)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

"""Linear Regression chart"""

import matplotlib.pyplot as plt

# Plotting Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, edgecolors=(0, 0, 0))
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Prices (Linear Regression)')
plt.show()

print("Linear Regression:")
print(f"Mean Absolute Error (MAE): {mae_lr}")
print(f"Mean Squared Error (MSE): {mse_lr}")
print(f"R² Score: {r2_lr}")

scores_lr = cross_val_score(model_lr, X, y, cv=5, scoring='neg_mean_absolute_error')
print(f"Cross-validated MAE (Linear Regression): {-scores_lr.mean()}")

"""## **Random Forest**"""

param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

rf = RandomForestRegressor()
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")

best_rf = grid_search.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)

mae_best_rf = mean_absolute_error(y_test, y_pred_best_rf)
mse_best_rf = mean_squared_error(y_test, y_pred_best_rf)
r2_best_rf = r2_score(y_test, y_pred_best_rf)

print("\nRandom Forest Regressor:")
print(f"Mean Absolute Error (MAE): {mae_best_rf}")
print(f"Mean Squared Error (MSE): {mse_best_rf}")
print(f"R² Score: {r2_best_rf}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best_rf, edgecolors=(0, 0, 0))
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Prices (Decision Tree Regressor)')
plt.show()

mae_log = mae_best_rf
rata_rata_kesalahan_asli = np.expm1(mae_log)
print(f"Rata-rata kesalahan dalam skala asli: {rata_rata_kesalahan_asli*100}%")

"""## **CART**"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

param_grid = {
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}

dt = DecisionTreeRegressor(random_state=42)

grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")

best_dt = grid_search.best_estimator_
y_pred_best_dt = best_dt.predict(X_test)

mae_best_dt = mean_absolute_error(y_test, y_pred_best_dt)
mse_best_dt = mean_squared_error(y_test, y_pred_best_dt)
r2_best_dt = r2_score(y_test, y_pred_best_dt)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best_dt, edgecolors=(0, 0, 0))
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Prices (Decision Tree Regressor)')
plt.show()

print("Decision Tree Regressor:")
print(f"Mean Absolute Error (MAE): {mae_best_dt}")
print(f"Mean Squared Error (MSE): {mse_best_dt}")
print(f"R² Score: {r2_best_dt}")

"""### Kesimpulan

Model Decision Tree Regressor yang telah dilatih menunjukkan performa yang cukup baik berdasarkan metrik evaluasi yang digunakan:

- MAE yang rendah menunjukkan bahwa rata-rata kesalahan absolut cukup kecil, mengindikasikan prediksi yang cukup akurat.
- MSE yang rendah menunjukkan bahwa model juga berhasil meminimalkan kesalahan besar dalam prediksinya.
- R² yang tinggi (0.7702) menunjukkan bahwa model mampu menjelaskan sebagian besar variabilitas dalam data target, menunjukkan kecocokan yang kuat dengan data.

Secara keseluruhan, model ini cukup efektif dalam memprediksi nilai target dan dapat dianggap sebagai model yang handal untuk dataset yang digunakan.
"""

scores_dt = cross_val_score(best_dt, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
print(f"Cross-validated MAE (Decision Tree Regressor): {-scores_dt.mean()}")

"""### Kesimpulan

Dengan MAE sekitar 0.3306, model ini memiliki rata-rata kesalahan prediksi yang cukup kecil, yang menunjukkan prediksi yang cukup akurat.

Hasil cross-validated MAE yang rendah menunjukkan bahwa model Decision Tree Regressor konsisten dalam performanya di berbagai subset data yang berbeda selama proses cross-validation. Ini menunjukkan bahwa model tidak terlalu bergantung pada satu subset data tertentu untuk performa yang baik.
"""

mae_log = mae_best_dt
rata_rata_kesalahan_asli = np.expm1(mae_log)
print(f"Rata-rata kesalahan dalam skala asli: {rata_rata_kesalahan_asli * 100}%")

"""### Kesimpulan

Secara keseluruhan, meskipun ada kesalahan rata-rata sebesar 39.77% dalam skala asli, model ini tetap memberikan prediksi yang cukup akurat dan dapat diandalkan untuk sebagian besar aplikasi.

## **SVR**
"""

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5, 1],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1]
}

svr = SVR(kernel='rbf')
grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")

best_svr = grid_search.best_estimator_
y_pred = best_svr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nSVR (Tuned):")
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

plt.scatter(y_test, y_pred, color='blue')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('SVR Predictions vs Actual Prices')
plt.show()

"""Kesimpulan menggunakan SVM

## **XGBOOST**
"""

from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Parameter grid untuk tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7, 9],
    'colsample_bytree': [0.6, 0.8, 1.0],
}
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")

best_xgb = grid_search.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)


mae_xgb = mean_absolute_error(y_test, y_pred)
mse_xgb = mean_squared_error(y_test, y_pred)
r2_xgb = r2_score(y_test, y_pred)

print("\nXGBoost Regressor (Tuned):")
print(f"Mean Absolute Error (MAE): {mae_xgb}")
print(f"Mean Squared Error (MSE): {mse_xgb}")
print(f"R² Score: {r2_xgb}")


xgb.plot_importance(best_xgb)
plt.show()

"""### GridSearch XGBoost"""

# from sklearn.model_selection import GridSearchCV
# from xgboost import XGBClassifier, XGBRegressor
# parameters = {'nthread':[4],
#               'objective':['reg:absoluteerror'],
#               'learning_rate': [0.05],
#               'max_depth': [6, 7, 8],
#               'min_child_weight': [11],
#               'silent': [1],
#               'subsample': [0.8],
#               'colsample_bytree': [0.7],
#               'n_estimators': [5],
#               'missing':[-999],
#               'seed': [1337]};

# grid_xgb = GridSearchCV(estimator=model, param_grid=parameters, scoring='accuracy', cv=5, verbose=1, error_score='raise')
# grid_xgb.fit(X_train, y_train)
# best_params_xgb = grid_xgb.best_params_
# print(f"Best parameters for Decision Tree: {best_params_xgb}")

"""## **Ensemble Learning**

Weight Counting
"""

inv_mae_xgb = 1 / mae_xgb
inv_mae_svr = 1 / mae
inv_mae_dt = 1 / mae_best_dt
inv_mae_rf = 1 / mae_best_rf

total_inv_mae = inv_mae_xgb + inv_mae_svr + inv_mae_dt + inv_mae_rf
w_xgb = inv_mae_xgb / total_inv_mae
w_svr = inv_mae_svr / total_inv_mae
w_dt = inv_mae_dt / total_inv_mae
w_rf = inv_mae_rf / total_inv_mae

print(f"Bobot XGB: {w_xgb}")
print(f"Bobot SVR: {w_svr}")
print(f"Bobot DT: {w_dt}")
print(f"Bobot RF: {w_rf}")

"""Lets counting"""

y_pred_ensemble_weighted = (w_xgb * y_pred_xgb + w_svr * y_pred + w_dt * y_pred_best_dt + w_rf * y_pred_best_rf)

mae_ensemble_weighted = mean_absolute_error(y_test, y_pred_ensemble_weighted)
mse_ensemble_weighted = mean_squared_error(y_test, y_pred_ensemble_weighted)
r2_ensemble_weighted = r2_score(y_test, y_pred_ensemble_weighted)

print(f"Weighted Ensemble Model - MAE: {mae_ensemble_weighted}")
print(f"Weighted Ensemble Model - MSE: {mse_ensemble_weighted}")
print(f"Weighted Ensemble Model - R²: {r2_ensemble_weighted}")



"""## Data export to .pkl"""

import joblib


joblib.dump(best_dt, 'model_dt.pkl')

joblib.dump(scaler, 'scaler.pkl')

joblib.dump(best_xgb, 'model_xgb.pkl')

joblib.dump(label_encoders, 'label_encoders.pkl')

joblib.dump(label_encoders, 'label_encoders.pkl')

joblib.dump(principal, 'pca.pkl')