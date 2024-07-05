from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model-model
model_dt = joblib.load('model_dt.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
# Load label encoders
label_encoders = joblib.load('label_encoders.pkl')



# Fitur yang diharapkan dalam urutan yang benar
expected_features = ['jumlah kamar tidur', 'jumlah kamar mandi', 'luas tanah (m2)', 'luas bangunan (m2)', 
                     'carport (mobil)', 'pasokan listrik (watt)', 'Kabupaten/Kota', 'kecamatan', 'kelurahan', 
                     'keamanan (ada/tidak)', 'taman (ada/tidak)', 'jarak dengan rumah sakit terdekat (km)', 
                     'jarak dengan sekolah terdekat (km)', 'jarak dengan tol terdekat (km)']

def encode_json(json_data):
    """
    Function to encode JSON data into a DataFrame with the correct format for the model.
    """
    df = pd.DataFrame(json_data, index=[0])
    
    # Encode categorical columns
    for column in ['Kabupaten/Kota', 'kecamatan', 'kelurahan', 'keamanan (ada/tidak)', 'taman (ada/tidak)']:
        if column in df.columns:
            try:
                df[column] = label_encoders[column].transform(df[column])
            except ValueError:
                df[column] = -1  # atau nilai yang sesuai jika label tidak dikenali
    
    # Convert numerical columns to float
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    
    # Ensure the DataFrame has the expected order of features
    df = df[expected_features]
    
    return df

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = encode_json(data)

    # Memastikan semua kolom ada
    for column in expected_features:
        if column not in df.columns:
            return jsonify({'error': f'Missing column: {column}'}), 400

    # Mengubah tipe data kolom numerik menjadi float
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    # Memastikan urutan kolom sesuai dengan urutan yang diharapkan
    df = df[expected_features]

    # Scale data
    X_scaled = scaler.transform(df)

    # PCA
    X_pca = pca.transform(X_scaled)
    # Prediksi dari masing-masing model
    pred_dt = model_dt.predict(X_pca)

    pred_dt = np.expm1(pred_dt)


    return jsonify({'predicted_price': pred_dt[0]})

if __name__ == '__main__':
    app.run(debug=True)
