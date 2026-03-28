"""
Pydantic models for BC Care Navigator API.
"""

from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional


class CareDestination(str, Enum):
    call_911 = "call_911"
    go_to_er = "go_to_er"
    upcc = "upcc"
    walk_in = "walk_in"
    pharmacist = "pharmacist"
    call_811 = "call_811"
    home_care = "home_care"


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    location: Optional[str] = None


class ExtractedSymptoms(BaseModel):
    symptoms: list[str]
    duration: Optional[str] = None
    severity: Optional[str] = None
    plain_descriptions: list[dict]  # [{"patient_words": "...", "clinical_term": "..."}]


class ChatResponse(BaseModel):
    reply: str
    ready_for_review: bool
    extracted_symptoms: Optional[ExtractedSymptoms] = None
    is_emergency: bool = False


class NavigationRequest(BaseModel):
    extracted_symptoms: ExtractedSymptoms
    location: str = "unknown"


class CommunityContext(BaseModel):
    chsa_name: str
    health_authority: str
    pct_without_family_doctor: float
    er_visits_per_1000: float
    opioid_overdose_rate: float
    pct_below_poverty_line: float


class NavigationResponse(BaseModel):
    destination: CareDestination
    destination_label: str
    is_emergency: bool
    headline: str
    reasoning: str
    what_to_bring: list[str]
    what_to_say: list[str]
    community_note: Optional[str] = None
    community_context: Optional[CommunityContext] = None
    safety_triggered: bool
    wait_times_context: Optional[str] = None
    healthlink_url: str = "https://www.healthlinkbc.ca/health-services"


class HealthCheckResponse(BaseModel):
    status: str
    version: str
    message: str
