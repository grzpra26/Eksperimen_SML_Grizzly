import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def kriteria_preprocessing(df_input):
    df_clean = df_input.copy()
    
    # 1. Menghapus data duplikat DAN langsung reset index di sini agar sinkron!
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    
    # 2. Encoding Data Kategorikal (Tetap sama)
    gender_map = {'Female': 0, 'Male': 1, 'Other': 2}
    smoking_map = {'No Info': 0, 'never': 1, 'former': 2, 'current': 3, 'not current': 4, 'ever': 5}
    df_clean['gender'] = df_clean['gender'].map(gender_map)
    df_clean['smoking_history'] = df_clean['smoking_history'].map(smoking_map)
    
    # 3. Memisahkan Fitur (X) dan Target (y)
    X = df_clean.drop(columns=['diabetes'])
    y = df_clean['diabetes']
    
    # 4. Standarisasi Fitur Numerik (Tetap sama)
    numerical_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # 5. Menggabungkan kembali
    df_ready = pd.concat([X, y], axis=1)
    
    return df_ready

if __name__ == "__main__":
    print("Memulai proses otomatisasi preprocessing...")
    
    # Membaca data mentah
    raw_data = pd.read_csv("diabetes_prediction_dataset.csv")
    
    # Menjalankan fungsi preprocessing
    clean_data = kriteria_preprocessing(raw_data)
    
    # Menyimpan hasil akhir ke folder kriteria 1
    clean_data.to_csv("preprocessing/namadataset_preprocessing.csv", index=False)
    
    print("Otomatisasi Selesai! File bersih berhasil disimpan.")