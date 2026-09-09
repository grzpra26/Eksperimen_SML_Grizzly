import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

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

# 1. Dataset & Training Model
data_path = "diabetes_preprocessing.csv"
df = pd.read_csv(data_path)

X = df.drop(columns=['diabetes'])
y = df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(random_state=42)
param_grid = {'n_estimators': [100, 150], 'max_depth': [10, 15], 'criterion': ['gini', 'entropy']}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Generasi Artefak Tambahan
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sehat', 'Diabetes'], yticklabels=['Sehat', 'Diabetes'])
plt.title("Confusion Matrix Performa Model")
plt.ylabel('Aktual')
plt.xlabel('Prediksi')
cf_matrix_path = "screenshot_confusion_matrix.png"
plt.savefig(cf_matrix_path, bbox_inches='tight')
plt.close()

report_dict = classification_report(y_test, y_pred, output_dict=True)
report_json_path = "classification_report_detail.json"
with open(report_json_path, "w") as json_file:
    json.dump(report_dict, json_file, indent=4)

# --- STEP 1: LOGGING KE LOCAL MLRUNS ---
print("Simpan ke Local mlruns...")
local_path = os.path.abspath("./mlruns")
os.environ["MLFLOW_TRACKING_URI"] = f"file:///{local_path}"
mlflow.set_tracking_uri(f"file:///{local_path}")
mlflow.set_experiment("Diabetes_Classification_Grizzly")

# Hapus ID Run bawaan mlflow run agar tidak crash di local
os.environ.pop("MLFLOW_RUN_ID", None)
os.environ.pop("MLFLOW_EXPERIMENT_ID", None)

with mlflow.start_run(run_name="Random_Forest_Hyperparameter_Tuning"):
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("model_algorithm", "RandomForestClassifier")
    mlflow.log_param("total_training_data", len(X_train))
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    
    mlflow.sklearn.log_model(best_model, "model")
    mlflow.log_artifact(cf_matrix_path)
    mlflow.log_artifact(report_json_path)

# --- STEP 2: LOGGING KE DAGSHUB ONLINE ---
print("Kirim ke DagsHub Online...")
dagshub.init(repo_owner='grzpra26', repo_name='Eksperimen_SML_Grizzly', mlflow=True)
dagshub_uri = mlflow.get_tracking_uri()
os.environ["MLFLOW_TRACKING_URI"] = dagshub_uri
mlflow.set_tracking_uri(dagshub_uri)
mlflow.set_experiment("Diabetes_Classification_Grizzly")

os.environ.pop("MLFLOW_RUN_ID", None)
os.environ.pop("MLFLOW_EXPERIMENT_ID", None)

with mlflow.start_run(run_name="Random_Forest_Hyperparameter_Tuning"):
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("model_algorithm", "RandomForestClassifier")
    mlflow.log_param("total_training_data", len(X_train))
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    
    mlflow.sklearn.log_model(best_model, "model")
    mlflow.log_artifact(cf_matrix_path)
    mlflow.log_artifact(report_json_path)

print("Proses Berhasil! Folder 'model' lokal beserta fits-nya telah dibuat! 🚀")
