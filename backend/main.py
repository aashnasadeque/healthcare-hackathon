"""
Patient Risk Dashboard — FastAPI Backend
Track 1: Clinical AI | UVic Healthcare Hackathon 2026
"""

import asyncio
import json
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional

import anthropic
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DATA_DIR = os.getenv("DATA_DIR", "data")
APP_VERSION = "2.0.0"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class RiskFlag(BaseModel):
    label: str
    severity: str  # "high" | "medium" | "low"

class PatientSummary(BaseModel):
    patient_id: str
    first_name: str
    last_name: str
    age: int
    sex: str
    risk_score: int
    risk_level: str  # "high" | "medium" | "low"
    top_flags: list[str]
    primary_language: str

class EncounterItem(BaseModel):
    encounter_id: str
    encounter_date: str
    encounter_type: str
    facility: str
    chief_complaint: str
    diagnosis_description: str
    triage_level: int
    disposition: str

class MedicationItem(BaseModel):
    drug_name: str
    dosage: str
    frequency: str
    prescriber: str
    start_date: str
    active: bool

class LabItem(BaseModel):
    test_name: str
    value: str
    unit: str
    reference_range: str
    abnormal_flag: str
    collected_date: str

class VitalsItem(BaseModel):
    heart_rate: float
    systolic_bp: float
    diastolic_bp: float
    temperature_celsius: float
    respiratory_rate: float
    o2_saturation: float
    pain_scale: float
    recorded_at: str

class PatientDetail(BaseModel):
    patient_id: str
    first_name: str
    last_name: str
    date_of_birth: str
    age: int
    sex: str
    postal_code: str
    primary_language: str
    blood_type: str
    risk_score: int
    risk_level: str
    flags: list[RiskFlag]
    ai_summary: str
    recent_encounters: list[EncounterItem]
    active_medications: list[MedicationItem]
    recent_labs: list[LabItem]
    latest_vitals: Optional[VitalsItem]

class HealthCheckResponse(BaseModel):
    status: str
    version: str

# ---------------------------------------------------------------------------
# Risk scoring — pure Python, no AI
# ---------------------------------------------------------------------------

def compute_risk(patient_id: str, encounters_df, medications_df, labs_df, vitals_df, age: int) -> tuple[int, list[RiskFlag]]:
    score = 0
    flags: list[RiskFlag] = []
    cutoff = datetime.now() - timedelta(days=365)

    # --- Emergency encounters in last 12 months ---
    p_enc = encounters_df[encounters_df["patient_id"] == patient_id].copy()
    p_enc["encounter_date"] = pd.to_datetime(p_enc["encounter_date"], errors="coerce")
    recent_enc = p_enc[p_enc["encounter_date"] >= cutoff]
    er_visits = recent_enc[recent_enc["encounter_type"] == "emergency"]
    if len(er_visits) > 0:
        score += len(er_visits) * 6
        flags.append(RiskFlag(label=f"{len(er_visits)} ER visit(s) in past year", severity="high"))

    # High triage level (1 or 2 = most urgent)
    critical_triage = recent_enc[recent_enc["triage_level"].isin([1, 2])]
    if len(critical_triage) > 0:
        score += len(critical_triage) * 3
        flags.append(RiskFlag(label=f"{len(critical_triage)} critical triage encounter(s)", severity="high"))

    # --- Polypharmacy ---
    p_meds = medications_df[
        (medications_df["patient_id"] == patient_id) &
        (medications_df["active"].astype(str).str.lower() == "true")
    ]
    med_count = len(p_meds)
    if med_count >= 10:
        score += 10
        flags.append(RiskFlag(label=f"High polypharmacy: {med_count} active medications", severity="high"))
    elif med_count >= 5:
        score += 5
        flags.append(RiskFlag(label=f"Polypharmacy: {med_count} active medications", severity="medium"))

    # --- Abnormal labs ---
    p_labs = labs_df[labs_df["patient_id"] == patient_id].copy()
    p_labs["collected_date"] = pd.to_datetime(p_labs["collected_date"], errors="coerce")
    recent_labs = p_labs[p_labs["collected_date"] >= cutoff]
    abnormal_labs = recent_labs[recent_labs["abnormal_flag"].str.upper().isin(["H", "L", "A", "HH", "LL", "Y"])]
    if len(abnormal_labs) > 0:
        score += len(abnormal_labs) * 3
        tests = ", ".join(abnormal_labs["test_name"].unique()[:3])
        flags.append(RiskFlag(label=f"{len(abnormal_labs)} abnormal lab(s): {tests}", severity="medium"))

    # --- Vitals ---
    p_vitals = vitals_df[vitals_df["patient_id"] == patient_id].copy()
    if not p_vitals.empty:
        p_vitals["recorded_at"] = pd.to_datetime(p_vitals["recorded_at"], errors="coerce")
        latest = p_vitals.sort_values("recorded_at").iloc[-1]
        try:
            o2 = float(latest["o2_saturation"])
            if o2 < 92:
                score += 8
                flags.append(RiskFlag(label=f"Critical O2 sat: {o2}%", severity="high"))
            elif o2 < 95:
                score += 4
                flags.append(RiskFlag(label=f"Low O2 sat: {o2}%", severity="medium"))
        except (ValueError, TypeError):
            pass
        try:
            sbp = float(latest["systolic_bp"])
            if sbp >= 180:
                score += 8
                flags.append(RiskFlag(label=f"Hypertensive crisis: {sbp} mmHg systolic", severity="high"))
            elif sbp >= 140:
                score += 3
                flags.append(RiskFlag(label=f"High BP: {sbp} mmHg systolic", severity="medium"))
        except (ValueError, TypeError):
            pass
        try:
            hr = float(latest["heart_rate"])
            if hr > 120 or hr < 45:
                score += 6
                flags.append(RiskFlag(label=f"Abnormal heart rate: {hr} bpm", severity="high"))
            elif hr > 100 or hr < 55:
                score += 3
                flags.append(RiskFlag(label=f"Borderline heart rate: {hr} bpm", severity="medium"))
        except (ValueError, TypeError):
            pass
        try:
            temp = float(latest["temperature_celsius"])
            if temp >= 39.5:
                score += 5
                flags.append(RiskFlag(label=f"High fever: {temp}°C", severity="high"))
            elif temp >= 38.5:
                score += 2
                flags.append(RiskFlag(label=f"Fever: {temp}°C", severity="medium"))
        except (ValueError, TypeError):
            pass

    # --- Age ---
    if age >= 80:
        score += 8
        flags.append(RiskFlag(label="Age 80+: elevated baseline risk", severity="medium"))
    elif age >= 65:
        score += 4
        flags.append(RiskFlag(label="Age 65+: elevated baseline risk", severity="low"))

    return score, flags


def risk_level(score: int) -> str:
    if score >= 20:
        return "high"
    elif score >= 10:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Startup — precompute risk scores
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[startup] Patient Risk Dashboard v{APP_VERSION}")

    try:
        patients = pd.read_csv(os.path.join(DATA_DIR, "patients.csv"))
        encounters = pd.read_csv(os.path.join(DATA_DIR, "encounters.csv"))
        medications = pd.read_csv(os.path.join(DATA_DIR, "medications.csv"))
        labs = pd.read_csv(os.path.join(DATA_DIR, "lab_results.csv"))
        vitals = pd.read_csv(os.path.join(DATA_DIR, "vitals.csv"))
        print(f"[startup] Loaded: {len(patients)} patients, {len(encounters)} encounters, "
              f"{len(medications)} medications, {len(labs)} labs, {len(vitals)} vitals")
    except Exception as e:
        print(f"[startup] ERROR loading data: {e}")
        patients = encounters = medications = labs = vitals = pd.DataFrame()

    app.state.patients = patients
    app.state.encounters = encounters
    app.state.medications = medications
    app.state.labs = labs
    app.state.vitals = vitals

    # Precompute risk scores for all patients
    ranked = []
    if not patients.empty:
        for _, row in patients.iterrows():
            pid = row["patient_id"]
            age = int(row.get("age", 0))
            score, flags = compute_risk(pid, encounters, medications, labs, vitals, age)
            ranked.append({
                "patient_id": pid,
                "first_name": str(row.get("first_name", "")),
                "last_name": str(row.get("last_name", "")),
                "age": age,
                "sex": str(row.get("sex", "")),
                "primary_language": str(row.get("primary_language", "English")),
                "risk_score": score,
                "risk_level": risk_level(score),
                "top_flags": [f.label for f in flags[:3]],
            })
        ranked.sort(key=lambda x: x["risk_score"], reverse=True)
        print(f"[startup] Risk scores computed for {len(ranked)} patients")

    app.state.ranked_patients = ranked
    yield
    print("[shutdown] Patient Risk Dashboard shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Patient Risk Dashboard", version=APP_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Claude helper
# ---------------------------------------------------------------------------

def _call_claude_sync(system: str, user: str, max_tokens: int = 400) -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


async def _call_claude(system: str, user: str, max_tokens: int = 400) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: _call_claude_sync(system, user, max_tokens))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthCheckResponse)
async def health():
    return HealthCheckResponse(status="ok", version=APP_VERSION)


@app.get("/api/patients", response_model=list[PatientSummary])
async def get_patients(limit: int = 50, search: str = ""):
    ranked = app.state.ranked_patients
    if search:
        s = search.lower()
        ranked = [p for p in ranked if s in p["first_name"].lower() or s in p["last_name"].lower() or s in p["patient_id"].lower()]
    return [PatientSummary(**p) for p in ranked[:limit]]


@app.get("/api/patient/{patient_id}", response_model=PatientDetail)
async def get_patient(patient_id: str):
    patients_df = app.state.patients
    encounters_df = app.state.encounters
    medications_df = app.state.medications
    labs_df = app.state.labs
    vitals_df = app.state.vitals

    # Find patient row
    p_rows = patients_df[patients_df["patient_id"] == patient_id]
    if p_rows.empty:
        raise HTTPException(status_code=404, detail="Patient not found")
    p = p_rows.iloc[0]
    age = int(p.get("age", 0))

    # Recompute risk (already cached but we need flags too)
    score, flags = compute_risk(patient_id, encounters_df, medications_df, labs_df, vitals_df, age)

    # Recent encounters (last 10)
    p_enc = encounters_df[encounters_df["patient_id"] == patient_id].copy()
    p_enc["encounter_date"] = pd.to_datetime(p_enc["encounter_date"], errors="coerce")
    p_enc = p_enc.sort_values("encounter_date", ascending=False).head(10)
    recent_encounters = [
        EncounterItem(
            encounter_id=str(r["encounter_id"]),
            encounter_date=str(r["encounter_date"])[:10],
            encounter_type=str(r["encounter_type"]),
            facility=str(r["facility"]),
            chief_complaint=str(r["chief_complaint"]),
            diagnosis_description=str(r["diagnosis_description"]),
            triage_level=int(r["triage_level"]) if pd.notna(r["triage_level"]) else 5,
            disposition=str(r["disposition"]),
        )
        for _, r in p_enc.iterrows()
    ]

    # Active medications
    p_meds = medications_df[
        (medications_df["patient_id"] == patient_id) &
        (medications_df["active"].astype(str).str.lower() == "true")
    ]
    active_medications = [
        MedicationItem(
            drug_name=str(r["drug_name"]),
            dosage=str(r["dosage"]),
            frequency=str(r["frequency"]),
            prescriber=str(r["prescriber"]),
            start_date=str(r["start_date"]),
            active=True,
        )
        for _, r in p_meds.iterrows()
    ]

    # Recent labs (last 15, sorted by date)
    p_labs = labs_df[labs_df["patient_id"] == patient_id].copy()
    p_labs["collected_date"] = pd.to_datetime(p_labs["collected_date"], errors="coerce")
    p_labs = p_labs.sort_values("collected_date", ascending=False).head(15)
    recent_labs = [
        LabItem(
            test_name=str(r["test_name"]),
            value=str(r["value"]),
            unit=str(r["unit"]),
            reference_range=f"{r['reference_range_low']}–{r['reference_range_high']}",
            abnormal_flag=str(r["abnormal_flag"]),
            collected_date=str(r["collected_date"])[:10],
        )
        for _, r in p_labs.iterrows()
    ]

    # Latest vitals
    latest_vitals = None
    p_vitals = vitals_df[vitals_df["patient_id"] == patient_id].copy()
    if not p_vitals.empty:
        p_vitals["recorded_at"] = pd.to_datetime(p_vitals["recorded_at"], errors="coerce")
        lv = p_vitals.sort_values("recorded_at").iloc[-1]
        latest_vitals = VitalsItem(
            heart_rate=float(lv["heart_rate"]),
            systolic_bp=float(lv["systolic_bp"]),
            diastolic_bp=float(lv["diastolic_bp"]),
            temperature_celsius=float(lv["temperature_celsius"]),
            respiratory_rate=float(lv["respiratory_rate"]),
            o2_saturation=float(lv["o2_saturation"]),
            pain_scale=float(lv["pain_scale"]),
            recorded_at=str(lv["recorded_at"]),
        )

    # AI clinical summary
    flag_text = "; ".join([f.label for f in flags]) if flags else "No critical flags"
    med_list = ", ".join([m.drug_name for m in active_medications[:8]]) or "none"
    recent_dx = ", ".join([e.diagnosis_description for e in recent_encounters[:3]]) or "none"
    vitals_text = ""
    if latest_vitals:
        vitals_text = (f"Latest vitals: HR {latest_vitals.heart_rate}, "
                       f"BP {latest_vitals.systolic_bp}/{latest_vitals.diastolic_bp}, "
                       f"O2 {latest_vitals.o2_saturation}%, Temp {latest_vitals.temperature_celsius}°C")

    ai_system = (
        "You are a clinical decision support assistant. Write a concise 3-sentence clinical summary "
        "for a clinician reviewing this patient. Focus on: (1) key risk factors, (2) what needs attention now, "
        "(3) one suggested next step. Be specific, clinical, and direct. Do not diagnose. Plain text only, no markdown."
    )
    ai_user = (
        f"Patient: {p['first_name']} {p['last_name']}, {age}yo {p['sex']}\n"
        f"Risk score: {score} ({risk_level(score)})\n"
        f"Risk flags: {flag_text}\n"
        f"Active medications ({len(active_medications)}): {med_list}\n"
        f"Recent diagnoses: {recent_dx}\n"
        f"{vitals_text}\n\n"
        "Write the 3-sentence clinical summary now."
    )

    try:
        ai_summary = await _call_claude(ai_system, ai_user, max_tokens=300)
    except Exception as e:
        print(f"[patient] Claude failed: {e}")
        ai_summary = (
            f"{p['first_name']} {p['last_name']} has a risk score of {score} ({risk_level(score)} risk). "
            f"Key concerns: {flag_text}. "
            f"Review active medications and recent encounters for follow-up."
        )

    return PatientDetail(
        patient_id=patient_id,
        first_name=str(p["first_name"]),
        last_name=str(p["last_name"]),
        date_of_birth=str(p["date_of_birth"]),
        age=age,
        sex=str(p["sex"]),
        postal_code=str(p["postal_code"]),
        primary_language=str(p["primary_language"]),
        blood_type=str(p["blood_type"]),
        risk_score=score,
        risk_level=risk_level(score),
        flags=flags,
        ai_summary=ai_summary,
        recent_encounters=recent_encounters,
        active_medications=active_medications,
        recent_labs=recent_labs,
        latest_vitals=latest_vitals,
    )


# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------

_frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    PORT = int(os.getenv("PORT", "8000"))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=PORT, reload=True)
