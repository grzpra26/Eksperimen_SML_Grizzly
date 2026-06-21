import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Inisialisasi ke DagsHub
dagshub.init(
    repo_owner='grzpra26', 
    repo_name='Eksperimen_SML_Grizzly', 
    mlflow=True
)

# 2. Mengaktifkan Autolog (Syarat Wajib Kriteria Basic untuk modelling.py)
mlflow.autolog()

# 3. Memuat Dataset Preprocessing
data_path = "diabetes_preprocessing.csv"
df = pd.read_csv(data_path)

X = df.drop(columns=['diabetes'])
y = df['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Setup Eksperimen
mlflow.set_experiment("Diabetes_Classification_Grizzly")

with mlflow.start_run(run_name="Baseline_Random_Forest"):
    # Model sederhana tanpa Hyperparameter Tuning
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    print("Baseline Model berhasil dilatih menggunakan Autolog! 🚀")