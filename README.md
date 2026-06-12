# Real-Time Fraud Detection API

A production-style ML micro-service that detects fraudulent credit-card transactions in real time using a Random Forest classifier served via FastAPI.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | `scikit-learn` RandomForestClassifier (balanced class weights) |
| API   | `FastAPI` + `Pydantic` v2 |
| Server| `Uvicorn` (ASGI) |
| Artefact serialisation | `joblib` |

---

## Quickstart

### 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### 2 — Train the model
```bash
python model.py
```
This will:
- Generate 50 000 synthetic transactions (~1 % fraud)
- Train a Random Forest with `class_weight='balanced'`
- Print ROC-AUC and a full classification report
- Save the artefact to **`fraud_model.pkl`**

### 3 — Start the API server
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 4 — Test the API

**Interactive docs (Swagger UI):**  
Open `http://127.0.0.1:8000/docs` in your browser.

**cURL – single prediction:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "amount": 450.00, "hour_of_day": 2, "day_of_week": 6,
       "v1": -3.1, "v2": 1.2, "v3": -2.8, "v4": 3.1, "v5": -1.6,
       "v6": 2.4, "v7": -0.9, "v8": 0.3, "v9": -1.1, "v10": 1.7,
       "v11": -0.5, "v12": 0.8, "v13": -1.3, "v14": 0.6, "v15": -0.7,
       "v16": 1.0, "v17": -0.4
     }'
```

**Expected response:**
```json
{
  "is_fraud": true,
  "fraud_probability": 0.873,
  "confidence": "High"
}
```

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Health ping |
| `GET`  | `/health` | Model-loaded status |
| `POST` | `/predict` | Single-transaction inference |
| `POST` | `/predict/batch` | Batch inference (≤ 1 000 rows) |

---

## Project Structure
```
Fraud_Detection_API/
├── model.py          # Dataset generation, training, evaluation, export
├── app.py            # FastAPI application & inference endpoints
├── requirements.txt  # Pinned dependencies
├── README.md         # This file
└── fraud_model.pkl   # Generated after running model.py
```
