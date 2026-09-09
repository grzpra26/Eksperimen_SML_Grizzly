import os
os.environ.pop("MLFLOW_RUN_ID", None)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# 1. Inisialisasi Koneksi ke DagsHub Online Server
dagshub.init(
    repo_owner='grzpra26', 
    repo_name='Eksperimen_SML_Grizzly', 
    mlflow=True
)

# 2. Memuat Dataset Hasil Preprocessing Kriteria 1
# Memastikan membaca file dataset bersih hasil olahan sebelumnya
data_path = "diabetes_preprocessing.csv"
df = pd.read_csv(data_path)

X = df.drop(columns=['diabetes'])
y = df['diabetes']

# Membagi data menjadi Training dan Testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Setup Nama Eksperimen di MLflow
mlflow.set_experiment("Diabetes_Classification_Grizzly")

# 4. Memulai Proses Pelatihan Model dan Tuning
with mlflow.start_run(run_name="Random_Forest_Hyperparameter_Tuning"):
    
    # Inisialisasi base model
    rf = RandomForestClassifier(random_state=42)
    
    # Parameter yang akan diuji dalam proses tuning
    param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [10, 15],
    'criterion': ['gini', 'entropy']
}
    
    # Melakukan Pencarian Parameter Terbaik (Hyperparameter Tuning)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Mengambil model terbaik hasil tuning
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # --- MANUAL LOGGING PARAMETER (Syarat Skilled/Advance) ---
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("model_algorithm", "RandomForestClassifier")
    mlflow.log_param("total_training_data", len(X_train))
    
    # --- MANUAL LOGGING METRIK (Syarat Skilled/Advance) ---
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    
    # --- SAVE & LOG MODEL ---
    mlflow.sklearn.log_model(best_model, "diabetes_rf_model")
    
    # --- LOG MINIMAL 2 ARTEFAK TAMBAHAN (Syarat Mutlak Advance) ---
    # Artefak Tambahan 1: Visualisasi Grafik Confusion Matrix (.png)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sehat', 'Diabetes'], yticklabels=['Sehat', 'Diabetes'])
    plt.title("Confusion Matrix Performa Model")
    plt.ylabel('Aktual')
    plt.xlabel('Prediksi')
    
    cf_matrix_path = "screenshot_confusion_matrix.png"
    plt.savefig(cf_matrix_path, bbox_inches='tight')
    plt.close()
    
    # Kirim file gambar ke DagsHub Artifact Storage
    mlflow.log_artifact(cf_matrix_path)
    
    # Artefak Tambahan 2: Detail Laporan Klasifikasi Lengkap (.json)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_json_path = "classification_report_detail.json"
    
    with open(report_json_path, "w") as json_file:
        json.dump(report_dict, json_file, indent=4)
        
    # Kirim file JSON ke DagsHub Artifact Storage
    mlflow.log_artifact(report_json_path)

    print("Proses Pelatihan Selesai! Parameter, Metrik, dan 2 Artefak Tambahan Sukses Terkirim Ke DagsHub secara Online! 🚀")
