import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def kriteria_preprocessing(df_input):
    df_clean = df_input.copy()
    
    # 1. Drop Duplicates
    df_clean = df_clean.drop_duplicates()
    
    # 2. Manual Mapping Data Kategorikal
    gender_map = {'Female': 0, 'Male': 1, 'Other': 2}
    smoking_map = {'No Info': 0, 'never': 1, 'former': 2, 'current': 3, 'not current': 4, 'ever': 5}
    df_clean['gender'] = df_clean['gender'].map(gender_map)
    df_clean['smoking_history'] = df_clean['smoking_history'].map(smoking_map)
    
    # 3. Separate Features and Target
    X = df_clean.drop(columns=['diabetes'])
    y = df_clean['diabetes']
    
    # 4. Standard Scaling
    numerical_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # 5. Combine Back
    df_ready = pd.concat([X, y.reset_index(drop=True)], axis=1)
    return df_ready

# INI STRUKTUR UTAMA YANG DIMAKSUD (Fungsi Otomatisasi saat File Dijalankan)
if __name__ == "__main__":
    print("Memulai proses otomatisasi preprocessing...")
    
    # Membaca data mentah
    raw_data = pd.read_csv("diabetes_prediction_dataset.csv")
    
    # Menjalankan fungsi preprocessing
    clean_data = kriteria_preprocessing(raw_data)
    
    # Menyimpan hasil akhir ke folder kriteria 1
    clean_data.to_csv("preprocessing/namadataset_preprocessing.csv", index=False)
    
    print("Otomatisasi Selesai! File bersih berhasil disimpan.")