from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data
from data import load_data

x, y, target_names = load_data()
app = FastAPI()

class HealthData(BaseModel):
    bmi: float
    cholesterol: float
    blood_pressure: float

class HealthResponse(BaseModel):
    response: int

class HealthNameResponse(BaseModel):
    status: str

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict_name", response_model=HealthNameResponse)
async def predict_health_name(health_features: HealthData):
    try:
        features = [[health_features.bmi,
                     health_features.cholesterol,
                     health_features.blood_pressure]]

        prediction = predict_data(features)
        prediction_idx = int(prediction[0])
        status_name = target_names[prediction_idx]
        return HealthNameResponse(status=status_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=HealthResponse)
async def predict_health(health_features: HealthData):
    try:
        features = [[health_features.bmi,
                     health_features.cholesterol,
                     health_features.blood_pressure]]

        prediction = predict_data(features)
        return HealthResponse(response=int(prediction[0]) + 1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
