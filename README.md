# SvaraAI Reply Classifier

A clean, minimal ML-powered email reply classification system that categorizes prospect responses into:
- **Positive**: Interested in meeting/demo
- **Negative**: Not interested / rejection  
- **Neutral**: Non-committal or irrelevant

##  Minimal Project Structure

```
SvaraAI-Reply-Classifier/
├── app.py                     # FastAPI application
├── train_and_eval.py          # Model training script
├── requirements.txt           # Essential dependencies
├── Dockerfile                # Container configuration
├── .gitignore                # Git ignore rules
├── data/
│   └── reply_classification_dataset.csv
├── models/                   # (Created after training)
└── docs/
    └── notebook.ipynb        # Analysis and visualization
```

##  Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model**:
   ```bash
   python train_and_eval.py
   ```

3. **Start the API server**:
   ```bash
   python -m uvicorn app:app --host 127.0.0.1 --port 8001
   ```

4. **Test the API**:
   ```bash
   curl -X POST "http://127.0.0.1:8001/predict" \
        -H "Content-Type: application/json" \
        -d '{"text": "Looking forward to the demo!"}'
   ```

##  API Endpoints

- `POST /predict` - Classify a single email reply
- `POST /predict-batch` - Classify multiple replies
- `GET /health` - Check API health
- `GET /model-info` - Get model information

##  Analysis

Open `docs/notebook.ipynb` in Jupyter to explore:
- Dataset analysis and visualization
- Model performance comparison
- Text characteristics by category

##  Production Ready

- High accuracy (99.5% F1 score)
- Docker containerization
- Error handling and logging
- Interactive API documentation

- 
- ## Video Link
[Download and watch the video](https://drive.google.com/file/d/1htdM7b5RN123mk-gf_EAgWxfN1FVS_3C/view?usp=drive_link)


