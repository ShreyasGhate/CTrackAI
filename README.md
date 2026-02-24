<div align="center">

# 🌱 CTrackAI

### AI-Powered Carbon Footprint Tracking for IoT Devices

[![CI/CD Pipeline](https://github.com/ShreyasGhate/CTrackAI/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/ShreyasGhate/CTrackAI/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.6-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Near real-time carbon footprint monitoring, anomaly detection, and forecasting — device-specific, explainable, and production-ready.*

[API Docs](#api-endpoints) • [Pipeline](#pipeline) • [Tech Stack](#-tech-stack) • [Setup](#-local-setup) • [Deploy](#-deployment)

</div>

---

## 📌 What is CTrackAI?

CTrackAI is an AI service that monitors energy consumption from IoT sensors every **15–20 minutes**, calculates real CO₂ emissions per device, detects anomalies, explains why they happened, and forecasts next-day/next-week carbon output.

It is **device-specific** — when a device like an `HP LaserJet Pro M404dn` is registered, a RAG (Retrieval-Augmented Generation) layer fetches its exact hardware specs (rated wattage, idle draw, duty cycle) and uses them as context for all downstream calculations. This makes anomaly thresholds, carbon math, severity scores, and explanations accurate to that exact model — not generic averages.

---

## ✨ Key Features

- 🔌 **Near Real-Time Processing** — sensor readings processed every 15–20 minutes via REST or MQTT
- 🧠 **RAG Layer (USP)** — device-specific context via ChromaDB + sentence-transformers
- ⚗️ **Carbon Math Engine** — CO₂ = Energy (kWh) × Emission Factor, India grid ≈ 0.82 kg CO₂/kWh
- 🕐 **Context Engine** — time, day, working hours, India holiday calendar awareness
- 🚨 **3-Layer Anomaly Detection** — Rule Engine + Isolation Forest + Autoencoder
- 🔴 **Severity Scoring** — Low / Medium / High / Critical with weighted formula
- 💬 **SHAP Explanations** — human-readable reason for every alert, no LLM needed
- 📈 **Prophet Forecasting** — next-day and next-week CO₂ estimates with confidence intervals
- 📋 **Full Traceability** — structured JSON logs via loguru, audit trail for every decision
- 🚀 **Versioned API** — FastAPI with graceful degradation, timeouts, and `/api/v1/` versioning

---

## 🏗️ Pipeline

```text
Sensor Input (15-20 min)
↓
Data Quality Handler ← flags/replaces bad readings
↓
RAG Layer ← fetches device specs (cached after first lookup)
↓
Carbon Math Engine ← CO₂ = Energy × Emission Factor
↓
Context Engine ← time, day, working hours, rolling baseline
↓
Anomaly Detection ← 3-layer hybrid (Rule + Isolation Forest + Autoencoder)
↓
┌─────────┴─────────┐
Normal         Anomaly
↓                 ↓
│         Severity Scoring ← weighted formula → LOW/MEDIUM/HIGH/CRITICAL
│                 ↓
│         Explanation ← SHAP + template engine → human sentence
│                 ↓
└─────────┬─────────┘
          ↓
Forecasting Module ← Prophet → next-day / next-week estimate
          ↓
Traceability & Logging ← loguru → SQLite (dev) / TimescaleDB (prod)
          ↓
Versioned API Output ← FastAPI /api/v1/ endpoints
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **API & Serving** | FastAPI, Uvicorn, Gunicorn, Pydantic |
| **Carbon Math** | Custom logic + TOU Emission factors |
| **Context Engine** | pandas, `holidays` library |
| **RAG Layer** | ChromaDB, sentence-transformers/all-MiniLM-L6-v2, LangChain |
| **Anomaly Detection** | scikit-learn (Isolation Forest), Rule DB, PyTorch (stubbed) |
| **Explainability** | SHAP + custom template engine |
| **Forecasting** | Prophet (fbprophet), statsmodels (ARIMA fallback), pmdarima |
| **Logging** | loguru → SQLite |
| **ML Tracking** | MLflow (ready) |
| **CI/CD** | GitHub Actions → Docker → Railway |

---

## 📡 API Endpoints

Interactive docs available at `/docs` (Swagger UI) when running.

### Ingestion
- `POST /api/v1/readings` → ingest single sensor reading
- `POST /api/v1/readings/batch` → ingest multiple readings at once
- `POST /api/v1/readings/buffer/flush` → flush buffered readings
- `GET /api/v1/readings/buffer/stats` → buffer status

### Pipeline
- `GET /api/v1/pipeline/stats` → pipeline performance stats

### Devices
- `GET /api/v1/devices/{id}/summary` → device CO₂ summary
- `GET /api/v1/devices/{id}/history` → historical readings
- `GET /api/v1/devices/{id}/forecast` → next-day/next-week forecast

### Anomalies
- `GET /api/v1/anomalies/recent` → recent anomaly alerts with explanations

### Health
- `GET /health` → service health check

### Example Request
```bash
curl -X POST http://localhost:8000/api/v1/readings \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "HP_LaserJet_M404dn",
    "wattage": 820,
    "timestamp": "2026-02-24T11:00:00+05:30"
  }'
```

### Example Response
```json
{
  "device_id": "HP_LaserJet_M404dn",
  "co2_kg": 0.34,
  "anomaly_detected": true,
  "severity": "CRITICAL",
  "severity_score": 0.87,
  "explanation": "Device drawing 820W (near max rated 650W) during off-hours on a weekend. Possible hardware fault or unauthorized usage.",
  "forecast": {
    "next_day_estimate_co2_kg": 12.4,
    "confidence_interval": [11.8, 13.1]
  },
  "pipeline_version": "v1.0.0",
  "processing_time_ms": 340,
  "degraded_components": []
}
```

---

## 🚀 Local Setup

### Prerequisites
- Python 3.11+
- Docker (optional for containerized run)

**1. Clone the repo**
```bash
git clone https://github.com/ShreyasGhate/CTrackAI.git
cd CTrackAI
```

**2. Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
# Install CPU-capable torch first
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**4. Run the server locally**
```bash
python main.py
```
*Visit http://localhost:8000/docs for interactive API documentation.*

**5. Run with Docker Compose**
```bash
docker compose up -d --build
```
*Docker handles all dependencies, SQLite storage volumes, and Gunicorn workers automatically.*

---

## 🧪 Running Tests
```bash
python -m tests.integration_test
```

---

## 🌍 Deployment

CTrackAI is deployed via Docker on **Railway** with automated CI/CD through **GitHub Actions**.

### CI/CD Flow
`git push` → Action triggers → Docker image built & cached → Pushed to `ghcr.io/ShreyasGhate/CTrackAI` → Railway auto-redeploys.

### Environment Variables 
*(Set these in Railway / `.env`)*
- `REGION=maharashtra`
- `EMISSION_FACTOR=0.727`
- `MQTT_ENABLED=false`

---

## 📊 Carbon Math

The core formula used:
```text
CO₂ (kg) = Energy (kWh) × Emission Factor (kg CO₂/kWh)

Where:
  Energy (kWh)  = Wattage (W) × Time (hrs) / 1000
  India Grid    ≈ 0.727 kg CO₂/kWh (Maharashtra)
```

**References:**
- GHG Protocol Calculation Methodology
- Frontiers in Energy Research — DOI: 10.3389/fenrg.2022.974365
- Indian CEA Baseline CO2 Emission Database

---

## 📁 Project Structure

```text
CTrackAI/
├── .github/workflows/          ← CI/CD pipelines
├── api/                        ← FastAPI routers & endpoints
├── config/                     ← Pydantic environment settings
├── data/                       ← SQLite DB and ChromaDB vectors (ignored in Git)
├── ingestion/                  ← REST & MQTT data handlers + aggregation
├── models/                     ← Shared Pydantic data schemas
├── pipeline/                   ← All core ML engines
│   ├── anomaly_detection.py
│   ├── carbon_math.py
│   ├── context_engine.py
│   ├── data_quality.py
│   ├── explanation_generator.py
│   ├── forecast_engine.py
│   ├── orchestrator.py
│   ├── severity_scoring.py
│   └── traceability.py
├── rag/                        ← Knowledge base and retrievers
├── synthetic/                  ← Test data generator
├── tests/                      ← Integration scripts
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── main.py
└── requirements.txt
```
