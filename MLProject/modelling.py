import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Inisialisasi DagsHub untuk ambil URI online
dagshub.init(
    repo_owner='grzpra26', 
    repo_name='Eksperimen_SML_Grizzly', 
    mlflow=True
)
dagshub_uri = mlflow.get_tracking_uri()

# 2. Memuat Dataset
data_path = "diabetes_preprocessing.csv"
df = pd.read_csv(data_path)

X = df.drop(columns=['diabetes'])
y = df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Path absolut ke folder mlruns lokal
local_path = os.path.abspath("./mlruns")

# --- STEP A: LOGGING KE LOCAL ---
print("Logging ke Local mlruns...")
os.environ["MLFLOW_TRACKING_URI"] = f"file:///{local_path}"
mlflow.set_tracking_uri(f"file:///{local_path}")
mlflow.autolog()
mlflow.set_experiment("Diabetes_Classification_Grizzly")

with mlflow.start_run(run_name="Baseline_Random_Forest"):
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

# --- STEP B: LOGGING KE DAGSHUB ---
print("Logging ke DagsHub Online...")
os.environ["MLFLOW_TRACKING_URI"] = dagshub_uri
mlflow.set_tracking_uri(dagshub_uri)
mlflow.autolog()
mlflow.set_experiment("Diabetes_Classification_Grizzly")

with mlflow.start_run(run_name="Baseline_Random_Forest"):
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

print("Baseline Model Sukses Dihasilkan di Local & DagsHub! 🚀")