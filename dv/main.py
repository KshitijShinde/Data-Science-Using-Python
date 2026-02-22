import logging
import json
import pickle
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, conlist, Field
from typing import List, Dict, Any

# --- Structured Logging Setup ---
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "name": record.name,
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "request_id"):
            log_record["request_id"] = record.request_id
        return json.dumps(log_record)

logger = logging.getLogger("api_logger")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(JSONFormatter())
logger.addHandler(console_handler)


# --- FastAPI Application ---
app = FastAPI(title="ML Model Inference API")

# Load model globally
model = None

@app.on_event("startup")
async def startup_event():
    global model
    logger.info("Initializing API and loading ML model...")
    try:
        with open("iris_model.pkl", "rb") as f:
            model = pickle.load(f)
        logger.info("ML model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load ML model: {e}")
        raise RuntimeError("Could not load model file.")

# --- Middleware for Logging ---
import uuid

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    logger.info(f"Incoming Request: {request.method} {request.url.path}", extra={"request_id": request_id, "client_ip": request.client.host})
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} | Process Time: {process_time:.4f}s", extra={"request_id": request_id, "status_code": response.status_code})
    return response

# --- Exception Handlers ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error_msg = f"Request Validation Error: {exc.errors()}"
    logger.warning(error_msg)
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid input format.", "errors": exc.errors()},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_msg = f"Unhandled Exception: {str(exc)}"
    logger.error(error_msg, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )

# --- Pydantic Schemas ---
class IrisPredictionRequest(BaseModel):
    sepal_length: float = Field(..., description="Sepal Length in cm", gt=0)
    sepal_width: float = Field(..., description="Sepal Width in cm", gt=0)
    petal_length: float = Field(..., description="Petal Length in cm", gt=0)
    petal_width: float = Field(..., description="Petal Width in cm", gt=0)

class IrisPredictionResponse(BaseModel):
    prediction_class: int = Field(..., description="Predicted class for the Iris flower (0, 1, or 2)")
    prediction_label: str = Field(..., description="Human-readable label: Setosa, Versicolor, Virginica")

IRIS_CLASS_NAMES = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

# --- Endpoints ---
@app.post("/predict", response_model=IrisPredictionResponse)
async def predict(request: IrisPredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    logger.info(f"Predict endpoint called with data: {request.dict()}")
    
    # Prepare data for model
    input_features = [[
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width
    ]]
    
    # Predict
    try:
        prediction = model.predict(input_features)[0]
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise Exception("Prediction inference failed.")
    
    pred_class = int(prediction)
    pred_label = IRIS_CLASS_NAMES.get(pred_class, "Unknown")
    
    logger.info(f"Prediction successful: {pred_label} (Class {pred_class})")
    
    return IrisPredictionResponse(
        prediction_class=pred_class,
        prediction_label=pred_label
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
