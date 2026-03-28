"""
Healthcare Hackathon — FastAPI Backend
======================================
Run with:
    uvicorn backend.main:app --reload --port 8000

Interactive API docs:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from backend.models import AnalysisResult, HealthCheckResponse, PatientData

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()  # reads .env if present

DEBUG = os.getenv("DEBUG", "true").lower() == "true"
PORT = int(os.getenv("PORT", "8000"))

# Parse allowed origins from env, fall back to permissive defaults for hacking
_raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:5500,http://localhost:5500",
)
ALLOWED_ORIGINS: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]

APP_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown logic. Load models or DB connections here."""
    print(f"[startup] Healthcare API v{APP_VERSION} | debug={DEBUG}")
    # TODO: Load your ML model here, e.g.:
    # app.state.model = joblib.load(os.getenv("MODEL_PATH", "model.joblib"))
    yield
    print("[shutdown] Cleaning up resources...")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Healthcare Hackathon API",
    description=(
        "Rapid-prototype API for health informatics. "
        "Accepts patient data and returns a risk score. "
        "Replace the placeholder logic in `/api/analyze` with your model."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"http://localhost:\d+",  # allow any localhost port in dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["Meta"],
    summary="Health check",
)
async def health_check() -> HealthCheckResponse:
    """Liveness probe — returns 200 if the service is up."""
    return HealthCheckResponse(
        status="ok",
        version=APP_VERSION,
        message="Healthcare API is running.",
    )


@app.post(
    "/api/analyze",
    response_model=AnalysisResult,
    tags=["Analysis"],
    summary="Analyze patient data",
    status_code=status.HTTP_200_OK,
)
async def analyze_patient(payload: PatientData) -> AnalysisResult:
    """
    Accept structured patient data and return a risk assessment.

    **This is a placeholder implementation.** Replace the scoring logic
    below with your actual ML model or rule-based algorithm.

    The endpoint currently applies a simple heuristic:
    - Counts abnormal vitals and lab flags
    - Returns a proportional risk score
    """
    flags: list[str] = []
    recommendations: list[str] = []
    feature_values: dict = {}

    # --- Vitals checks ---
    if payload.vitals:
        v = payload.vitals

        if v.heart_rate is not None:
            feature_values["heart_rate"] = v.heart_rate
            if v.heart_rate > 100:
                flags.append("Tachycardia (HR > 100 bpm)")
                recommendations.append("Evaluate for underlying cause of elevated heart rate.")
            elif v.heart_rate < 60:
                flags.append("Bradycardia (HR < 60 bpm)")

        if v.systolic_bp is not None:
            feature_values["systolic_bp"] = v.systolic_bp
            if v.systolic_bp >= 180:
                flags.append("Hypertensive crisis (SBP ≥ 180 mmHg)")
                recommendations.append("Immediate BP management required.")
            elif v.systolic_bp >= 140:
                flags.append("Stage 2 hypertension (SBP ≥ 140 mmHg)")

        if v.spo2 is not None:
            feature_values["spo2"] = v.spo2
            if v.spo2 < 90:
                flags.append("Critical hypoxia (SpO2 < 90%)")
                recommendations.append("Supplemental oxygen therapy indicated.")
            elif v.spo2 < 95:
                flags.append("Low oxygen saturation (SpO2 < 95%)")

        if v.temperature_c is not None:
            feature_values["temperature_c"] = v.temperature_c
            if v.temperature_c >= 38.5:
                flags.append("Fever (temp ≥ 38.5 °C)")
                recommendations.append("Investigate for infectious source.")
            elif v.temperature_c < 36.0:
                flags.append("Hypothermia (temp < 36.0 °C)")

        if v.respiratory_rate is not None:
            feature_values["respiratory_rate"] = v.respiratory_rate
            if v.respiratory_rate > 20:
                flags.append("Tachypnoea (RR > 20 breaths/min)")
            elif v.respiratory_rate < 12:
                flags.append("Bradypnoea (RR < 12 breaths/min)")

        bmi = v.bmi
        if bmi is not None:
            feature_values["bmi"] = bmi
            if bmi >= 30:
                flags.append(f"Obesity (BMI {bmi})")
            elif bmi < 18.5:
                flags.append(f"Underweight (BMI {bmi})")

    # --- Lab checks ---
    for lab in payload.labs:
        feature_values[f"lab_{lab.name}"] = lab.value
        if lab.flag in ("H", "HH", "L", "LL"):
            label = "HIGH" if lab.flag in ("H", "HH") else "LOW"
            flags.append(f"Abnormal {lab.name}: {lab.value} {lab.unit or ''} ({label})")

    # --- Diagnosis count ---
    feature_values["n_diagnoses"] = len(payload.diagnoses)
    if len(payload.diagnoses) >= 3:
        flags.append(f"Multimorbidity: {len(payload.diagnoses)} active diagnoses")

    # --- Age adjustment ---
    if payload.age is not None:
        feature_values["age"] = payload.age
        if payload.age >= 75:
            flags.append("Advanced age (≥ 75 years) — increased baseline risk")

    # -----------------------------------------------------------------------
    # TODO: Replace this naive heuristic with your trained model, e.g.:
    #
    #   features = build_feature_vector(payload)
    #   risk_score = float(app.state.model.predict_proba([features])[0, 1])
    # -----------------------------------------------------------------------
    max_possible_flags = 10
    raw_score = min(len(flags) / max_possible_flags, 1.0)

    if raw_score < 0.25:
        risk_label = "low"
    elif raw_score < 0.6:
        risk_label = "moderate"
    else:
        risk_label = "high"

    if not recommendations and risk_label in ("moderate", "high"):
        recommendations.append("Schedule follow-up within 48 hours.")

    debug_info = {}
    if DEBUG:
        debug_info = {
            "n_flags": len(flags),
            "raw_score": raw_score,
            "features_used": feature_values,
            "model_note": "Placeholder heuristic — replace with real model.",
        }

    return AnalysisResult(
        patient_id=payload.patient_id,
        risk_score=round(raw_score, 3),
        risk_label=risk_label,
        flags=flags,
        recommendations=recommendations,
        debug=debug_info,
    )


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=PORT, reload=DEBUG)
