"""
SvaraAI Reply Classifier API
Classify email replies as positive, negative, or neutral
"""

import os
from typing import List, Dict, Any, Optional
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import traceback
import json

# Initialize FastAPI app
app = FastAPI(
    title="SvaraAI Reply Classifier API",
    description="Classify email replies as positive, negative, or neutral",
    version="1.0.0"
)

# Global variables for model components
model = None
vectorizer = None
label_encoder = None

# Pydantic models for request/response
class TextRequest(BaseModel):
    text: str

class BatchTextRequest(BaseModel):
    texts: List[str]

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]

class BatchPredictionResponse(BaseModel):
    results: List[Dict[str, Any]]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vectorizer_loaded: bool
    label_encoder_loaded: bool

class ModelInfoResponse(BaseModel):
    model_type: str
    labels: List[str]
    feature_count: int
    transformer_available: bool

# Load model components
def load_model_components():
    """Load the trained model, vectorizer, and label encoder"""
    global model, vectorizer, label_encoder
    
    try:
        model_path = os.path.join("models", "best_baseline_model.pkl")
        vectorizer_path = os.path.join("models", "tfidf_vectorizer.pkl")
        label_encoder_path = os.path.join("models", "label_encoder.pkl")
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(" Loaded baseline model successfully")
        else:
            print("  Baseline model not found")
            
        if os.path.exists(vectorizer_path):
            vectorizer = joblib.load(vectorizer_path)
            print(" Loaded TF-IDF vectorizer successfully")
        else:
            print("  TF-IDF vectorizer not found")
            
        if os.path.exists(label_encoder_path):
            label_encoder = joblib.load(label_encoder_path)
            print(" Loaded label encoder successfully")
        else:
            print("  Label encoder not found")
            
    except Exception as e:
        print(f" Error loading model components: {e}")
        traceback.print_exc()

# Load model components on startup
@app.on_event("startup")
async def startup_event():
    """Load model components when the app starts"""
    load_model_components()

# Root endpoint with API information
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - returns API information"""
    # Return HTML content for browsers
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SvaraAI Reply Classifier API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #333; }
            .endpoint { background-color: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            code { background-color: #eee; padding: 2px 5px; border-radius: 3px; }
            .note { background-color: #e7f3ff; padding: 15px; margin: 20px 0; border-left: 4px solid #007acc; }
        </style>
    </head>
    <body>
        <h1>SvaraAI Reply Classifier API</h1>
        <p>Classify email replies as positive, negative, or neutral</p>
        
        <div class="endpoint">
            <h3>API Endpoints</h3>
            <ul>
                <li><strong>POST /predict</strong> - Classify a text reply</li>
                <li><strong>POST /predict-batch</strong> - Classify multiple text replies</li>
                <li><strong>GET /health</strong> - Check API health</li>
                <li><strong>GET /model-info</strong> - Get model information</li>
            </ul>
        </div>
        
        <div class="endpoint">
            <h3>Example Usage</h3>
            <p><strong>Request:</strong></p>
            <code>POST /predict</code>
            <pre>
{
  "text": "Looking forward to the demo!"
}
            </pre>
            <p><strong>Response:</strong></p>
            <pre>
{
  "label": "positive",
  "confidence": 0.79,
  "probabilities": {
    "negative": 0.14,
    "neutral": 0.07,
    "positive": 0.79
  }
}
            </pre>
        </div>
        
        <div class="note">
            <p><strong>Interactive Documentation:</strong> Visit <a href="/docs">/docs</a> for interactive API documentation and testing.</p>
            <p><strong>Programmatic Access:</strong> For JSON responses, set the <code>Accept</code> header to <code>application/json</code>.</p>
        </div>
    </body>
    </html>
    """

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        vectorizer_loaded=vectorizer is not None,
        label_encoder_loaded=label_encoder is not None
    )

# Model information endpoint
@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information"""
    if model is None or vectorizer is None or label_encoder is None:
        raise HTTPException(status_code=500, detail="Model components not loaded")
    
    return ModelInfoResponse(
        model_type=type(model).__name__,
        labels=label_encoder.classes_.tolist(),
        feature_count=vectorizer.get_feature_names_out().size,
        transformer_available=True
    )

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextRequest):
    """Classify a single text reply"""
    if model is None or vectorizer is None or label_encoder is None:
        raise HTTPException(status_code=500, detail="Model components not loaded")
    
    try:
        # Preprocess text
        text = request.text.lower().strip()
        
        # Vectorize text
        text_vector = vectorizer.transform([text])
        
        # Make prediction
        prediction = model.predict(text_vector)[0]
        probabilities = model.predict_proba(text_vector)[0]
        
        # Decode label
        label = label_encoder.inverse_transform([prediction])[0]
        
        # Get confidence (probability of predicted class)
        confidence = max(probabilities)
        
        # Create probabilities dictionary
        prob_dict = {}
        for i, class_label in enumerate(label_encoder.classes_):
            prob_dict[class_label] = float(probabilities[i])
        
        return PredictionResponse(
            label=label,
            confidence=float(confidence),
            probabilities=prob_dict
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Batch prediction endpoint
@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(texts: List[str]):
    """Classify multiple text replies"""
    if model is None or vectorizer is None or label_encoder is None:
        raise HTTPException(status_code=500, detail="Model components not loaded")
    
    results = []
    
    try:
        for i, text in enumerate(texts):
            try:
                # Preprocess text
                processed_text = text.lower().strip()
                
                # Vectorize text
                text_vector = vectorizer.transform([processed_text])
                
                # Make prediction
                prediction = model.predict(text_vector)[0]
                probabilities = model.predict_proba(text_vector)[0]
                
                # Decode label
                label = label_encoder.inverse_transform([prediction])[0]
                
                # Get confidence (probability of predicted class)
                confidence = max(probabilities)
                
                # Create probabilities dictionary
                prob_dict = {}
                for j, class_label in enumerate(label_encoder.classes_):
                    prob_dict[class_label] = float(probabilities[j])
                
                results.append({
                    "index": i,
                    "label": label,
                    "confidence": float(confidence),
                    "probabilities": prob_dict,
                    "error": None
                })
                
            except Exception as e:
                results.append({
                    "index": i,
                    "label": None,
                    "confidence": 0.0,
                    "probabilities": {},
                    "error": str(e)
                })
        
        return BatchPredictionResponse(results=results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")