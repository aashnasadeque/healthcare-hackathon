"""
Pydantic models for the Healthcare Hackathon API.

These define the shape of data coming in and going out of the API.
Extend or replace these with models that match your specific use case.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Sex(str, Enum):
    male = "male"
    female = "female"
    other = "other"
    unknown = "unknown"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class Vitals(BaseModel):
    """Common vital signs. All fields are optional — send what you have."""

    heart_rate: float | None = Field(None, description="Heart rate in bpm", ge=0, le=350)
    systolic_bp: float | None = Field(None, description="Systolic blood pressure (mmHg)", ge=0, le=350)
    diastolic_bp: float | None = Field(None, description="Diastolic blood pressure (mmHg)", ge=0, le=250)
    temperature_c: float | None = Field(None, description="Body temperature in Celsius", ge=25.0, le=45.0)
    spo2: float | None = Field(None, description="Oxygen saturation (%)", ge=0.0, le=100.0)
    respiratory_rate: float | None = Field(None, description="Respiratory rate (breaths/min)", ge=0, le=100)
    weight_kg: float | None = Field(None, description="Weight in kilograms", ge=0)
    height_cm: float | None = Field(None, description="Height in centimetres", ge=0)

    @property
    def bmi(self) -> float | None:
        """Body Mass Index, computed on the fly if height and weight are present."""
        if self.weight_kg and self.height_cm and self.height_cm > 0:
            return round(self.weight_kg / (self.height_cm / 100) ** 2, 1)
        return None


class LabResult(BaseModel):
    """A single lab result."""

    name: str = Field(..., description="Lab test name, e.g. 'HbA1c', 'eGFR'")
    value: float = Field(..., description="Numeric result value")
    unit: str | None = Field(None, description="Unit of measure, e.g. 'mmol/L'")
    reference_range: str | None = Field(None, description="Normal reference range, e.g. '3.5–5.0'")
    flag: str | None = Field(None, description="Interpretation flag: 'H', 'L', 'HH', 'LL', 'N'")


class Diagnosis(BaseModel):
    """A coded diagnosis (ICD-10 or SNOMED CT)."""

    code: str = Field(..., description="Diagnosis code, e.g. 'E11.9' (ICD-10) or '73211009' (SNOMED)")
    system: str = Field("ICD-10", description="Coding system: 'ICD-10', 'SNOMED-CT', 'ICD-11'")
    display: str | None = Field(None, description="Human-readable label for the code")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class PatientData(BaseModel):
    """
    Generic patient data payload accepted by /api/analyze.

    All clinical fields are optional — the endpoint should gracefully handle
    partial data. Add or remove fields to match your dataset.
    """

    # Demographics
    patient_id: str = Field(..., description="De-identified patient identifier")
    age: int | None = Field(None, description="Patient age in years", ge=0, le=150)
    sex: Sex | None = Field(None, description="Biological sex")

    # Clinical data
    vitals: Vitals | None = None
    labs: list[LabResult] = Field(default_factory=list)
    diagnoses: list[Diagnosis] = Field(default_factory=list)

    # Free-form chief complaint or notes (no PHI in a real system)
    chief_complaint: str | None = Field(None, max_length=500)

    # Arbitrary extra fields — useful during rapid prototyping
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Catch-all for any additional features your model needs",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "patient_id": "PT-00123",
                    "age": 67,
                    "sex": "female",
                    "vitals": {
                        "heart_rate": 92,
                        "systolic_bp": 145,
                        "diastolic_bp": 88,
                        "temperature_c": 37.2,
                        "spo2": 96.0,
                        "respiratory_rate": 18,
                        "weight_kg": 72,
                        "height_cm": 162,
                    },
                    "labs": [
                        {"name": "HbA1c", "value": 8.1, "unit": "%", "flag": "H"},
                        {"name": "eGFR", "value": 55, "unit": "mL/min/1.73m²"},
                    ],
                    "diagnoses": [
                        {"code": "E11.9", "system": "ICD-10", "display": "Type 2 diabetes mellitus, unspecified"},
                        {"code": "I10", "system": "ICD-10", "display": "Essential hypertension"},
                    ],
                    "chief_complaint": "Fatigue and increased thirst for 2 weeks",
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class AnalysisResult(BaseModel):
    """Response returned by /api/analyze."""

    patient_id: str
    risk_score: float = Field(..., description="Composite risk score between 0.0 and 1.0", ge=0.0, le=1.0)
    risk_label: str = Field(..., description="Human-readable risk category: 'low', 'moderate', 'high'")
    flags: list[str] = Field(default_factory=list, description="List of clinical flags raised during analysis")
    recommendations: list[str] = Field(default_factory=list, description="Suggested follow-up actions")
    model_version: str = Field("0.1.0-placeholder", description="Version of the model that produced this result")
    debug: dict[str, Any] = Field(default_factory=dict, description="Extra info returned when DEBUG=true")


class HealthCheckResponse(BaseModel):
    status: str
    version: str
    message: str
