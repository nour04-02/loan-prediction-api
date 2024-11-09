from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib

# Load the model parameters
model_params = joblib.load("model.pkl")
weights = model_params["weights"]
bias = model_params["bias"]

# Define the predict function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, weights, bias):
    logits = np.dot(X, weights) + bias
    return sigmoid(logits)

# Create a FastAPI app
app = FastAPI()

# Define input data model
class InputData(BaseModel):
    features: list[float]

@app.post("/predict")
async def make_prediction(data: InputData):
    try:
        X = np.array(data.features).reshape(1, -1)  # Reshape input to match model's expected format
        y_pred = predict(X, weights, bias)
        prediction = int(y_pred >= 0.5)  # Assuming 0.5 threshold for binary classification

        return {"prediction": prediction, "probability": y_pred.item()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
