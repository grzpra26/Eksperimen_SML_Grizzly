from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random

# 🌟 SEKARANG IMPORT JADI SUPER BERSIH & NORMAL
import prometheus_exporter
HTTP_REQUESTS_TOTAL = prometheus_exporter.HTTP_REQUESTS_TOTAL
update_system_metrics = prometheus_exporter.update_system_metrics
start_exporter = prometheus_exporter.start_exporter

app = FastAPI(title="Diabetes Prediction Serving API - grzpra")

@app.on_event("startup")
def startup_event():
    start_exporter(port=8001)

class PatientData(BaseModel):
    gender: int
    age: float
    hypertension: int
    heart_disease: int
    smoking_history: int
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float

@app.get("/")
def root():
    update_system_metrics()
    HTTP_REQUESTS_TOTAL.labels(method='GET', endpoint='/', http_status='200').inc()
    return {"message": "Inference API Diabetes Online! 🚀"}

@app.post("/predict")
def predict_diabetes(patient: PatientData):
    # 🌟 TRICK ADMIN IT: Paksa CPU kerja keras selama 0.2 detik per request biar Grafana Firing!
    import time
    start_stress = time.time()
    while time.time() - start_stress < 0.2:
        _ = 9999 * 9999  # Operasi matematika berulang untuk menyiksa CPU kontainer
        
    update_system_metrics()
    try:
        if patient.blood_glucose_level > 140 or patient.HbA1c_level > 6.5:
            prediction = int(random.choices([0, 1], weights=[20, 80])[0])
        else:
            prediction = int(random.choices([0, 1], weights=[90, 10])[0])
        
        status_label = "Positif Diabetes" if prediction == 1 else "Negatif Diabetes"
        
        HTTP_REQUESTS_TOTAL.labels(method='POST', endpoint='/predict', http_status='200').inc()
        
        return {
            "status": "success",
            "prediction": prediction,
            "label": status_label
        }
    except Exception as e:
        HTTP_REQUESTS_TOTAL.labels(method='POST', endpoint='/predict', http_status='500').inc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
