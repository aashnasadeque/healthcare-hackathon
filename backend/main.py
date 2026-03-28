"""
BC Care Navigator — FastAPI Backend
=====================================
Run with:
    uvicorn backend.main:app --reload --port 8000

Interactive API docs:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

import asyncio
import json
import os
import re
from contextlib import asynccontextmanager
from typing import Optional

import anthropic
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.models import (
    CareDestination,
    ChatRequest,
    ChatResponse,
    CommunityContext,
    ExtractedSymptoms,
    HealthCheckResponse,
    NavigationRequest,
    NavigationResponse,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DATA_DIR = os.getenv("DATA_DIR", "data")
APP_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Emergency keywords — checked on EVERY user message and symptom list
# ---------------------------------------------------------------------------

EMERGENCY_KEYWORDS = {
    "chest pain", "chest pressure", "chest tightness", "heart attack",
    "can't breathe", "cant breathe", "not breathing", "stopped breathing",
    "face drooping", "arm weakness", "slurred speech", "stroke",
    "unconscious", "unresponsive", "passed out", "fainted",
    "seizure", "convulsing",
    "overdose",
    "throat closing", "anaphylaxis", "epipen",
    "purple spots", "meningitis",
    "severe bleeding", "bleeding won't stop", "coughing blood",
    "thoughts of suicide", "want to kill myself", "end my life",
}

DESTINATION_LABELS = {
    CareDestination.call_911: "Call 911",
    CareDestination.go_to_er: "Go to the Emergency Room",
    CareDestination.upcc: "Urgent and Primary Care Centre (UPCC)",
    CareDestination.walk_in: "Walk-In Clinic",
    CareDestination.pharmacist: "Pharmacist",
    CareDestination.call_811: "Call 8-1-1 (HealthLink BC)",
    CareDestination.home_care: "Home Care / Self-Care",
}

# Postal code prefix → community name
POSTAL_CODE_MAP = {
    "v8w": "Greater Victoria",
    "v8r": "Saanich",
    "v5k": "Vancouver",
    "v6b": "Vancouver",
    "v2c": "Kamloops",
    "v1y": "Kelowna",
    "v2j": "Prince George",
    "v3l": "Surrey",
    "v1s": "Salmon Arm",
    "v0e": "Revelstoke",
    "v1a": "Golden",
    "v2a": "Penticton",
    "v1t": "Vernon",
    "v2h": "Merritt",
    "v0n": "Pemberton",
    "v8g": "Terrace",
    "v1g": "Dawson Creek",
    "v0c": "Fort St. John",
    "v0j": "Fort Nelson",
    "v2g": "Williams Lake",
}

# ---------------------------------------------------------------------------
# App lifecycle — load CSV data on startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load BC health datasets into app.state on startup."""
    print(f"[startup] BC Care Navigator API v{APP_VERSION}")

    try:
        bc_path = os.path.join(DATA_DIR, "bc_health_indicators.csv")
        app.state.bc_data = pd.read_csv(bc_path)
        print(f"[startup] Loaded bc_health_indicators.csv — {len(app.state.bc_data)} rows")
    except Exception as exc:
        print(f"[startup] WARNING: Could not load bc_health_indicators.csv: {exc}")
        app.state.bc_data = pd.DataFrame()

    try:
        wt_path = os.path.join(DATA_DIR, "wait_times_mock.csv")
        app.state.wait_times_data = pd.read_csv(wt_path)
        print(f"[startup] Loaded wait_times_mock.csv — {len(app.state.wait_times_data)} rows")
    except Exception as exc:
        print(f"[startup] WARNING: Could not load wait_times_mock.csv: {exc}")
        app.state.wait_times_data = pd.DataFrame()

    try:
        opioid_path = os.path.join(DATA_DIR, "opioid_harms_mock.csv")
        app.state.opioid_data = pd.read_csv(opioid_path)
        print(f"[startup] Loaded opioid_harms_mock.csv — {len(app.state.opioid_data)} rows")
    except Exception as exc:
        print(f"[startup] WARNING: Could not load opioid_harms_mock.csv: {exc}")
        app.state.opioid_data = pd.DataFrame()

    yield
    print("[shutdown] BC Care Navigator shutting down.")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="BC Care Navigator API",
    description=(
        "Conversational healthcare navigation for British Columbia. "
        "Guides patients to the right level of care based on symptoms and location."
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helper: emergency keyword detection
# ---------------------------------------------------------------------------


def _contains_emergency_keyword(text: str) -> bool:
    """Return True if text contains any emergency keyword as a whole phrase (case-insensitive)."""
    lower = text.lower()
    for kw in EMERGENCY_KEYWORDS:
        # Use word boundaries so "od" can't match inside "body", etc.
        if re.search(r'\b' + re.escape(kw) + r'\b', lower):
            return True
    return False


def _any_message_emergency(messages: list) -> bool:
    """Check all user messages for emergency keywords."""
    return any(
        _contains_emergency_keyword(m.content)
        for m in messages
        if m.role == "user"
    )


# ---------------------------------------------------------------------------
# Helper: community lookup (fuzzy match to bc_health_indicators.csv)
# ---------------------------------------------------------------------------


def lookup_community(location: str, bc_data: pd.DataFrame) -> Optional[dict]:
    """
    Fuzzy-match a location string to a row in bc_health_indicators.csv.
    Returns a dict of the matching row, or None if no match found.
    """
    if bc_data.empty or not location or location.strip().lower() in ("", "unknown"):
        return None

    # Normalize the input
    loc = location.lower().strip()
    # Strip common suffixes
    for strip_term in ["bc", "british columbia", ",", "."]:
        loc = loc.replace(strip_term, "").strip()

    if not loc:
        return None

    # Check for postal code prefix (first 3 chars)
    postal_prefix = loc.replace(" ", "")[:3]
    if postal_prefix in POSTAL_CODE_MAP:
        loc = POSTAL_CODE_MAP[postal_prefix].lower()

    # Normalise the chsa_name column for comparison
    names = bc_data["chsa_name"].str.lower().str.strip()

    # 1. Exact match
    exact = bc_data[names == loc]
    if not exact.empty:
        return exact.iloc[0].to_dict()

    # 2. Location is substring of chsa_name
    sub_fwd = bc_data[names.str.contains(re.escape(loc), na=False)]
    if not sub_fwd.empty:
        return sub_fwd.iloc[0].to_dict()

    # 3. chsa_name is substring of location (reverse)
    matches = bc_data[names.apply(lambda n: n in loc)]
    if not matches.empty:
        return matches.iloc[0].to_dict()

    return None


# ---------------------------------------------------------------------------
# Helper: determine care destination (pure Python rule tree)
# ---------------------------------------------------------------------------


def determine_destination(
    symptoms: list[str],
    severity: Optional[str],
    duration: Optional[str],
    community: Optional[dict],
) -> CareDestination:
    """Rule-based triage tree. Returns a CareDestination enum value."""
    symptom_set = {s.lower() for s in symptoms}
    text = " ".join(symptom_set)

    # --- Emergency / ER indicators ---
    er_terms = {
        "broken", "fracture", "sepsis", "pregnancy pain", "appendicitis",
        "severe abdominal", "eye injury", "head injury", "high fever confusion",
    }
    if any(t in text for t in er_terms):
        return CareDestination.go_to_er

    if severity and any(
        w in severity.lower()
        for w in ["severe", "very bad", "10", "9", "8", "worst", "unbearable", "now"]
    ):
        return CareDestination.go_to_er

    # --- UPCC ---
    upcc_terms = {
        "high fever", "fever", "stitches", "wound", "infection", "cellulitis",
        "mental health", "anxiety", "depression", "panic", "asthma",
        "breathing difficulty", "moderate pain", "sprain", "burn", "rash",
        "dehydration",
    }
    if any(t in text for t in upcc_terms):
        return CareDestination.upcc

    # --- Pharmacist (BC expanded scope) ---
    pharmacist_terms = {
        "uti", "urinary tract", "pink eye", "conjunctivitis",
        "refill", "prescription refill", "shingles", "hemorrhoids",
    }
    if any(t in text for t in pharmacist_terms):
        return CareDestination.pharmacist

    # --- Walk-in / equity escalation ---
    walkin_terms = {
        "cold", "flu", "ear infection", "sore throat", "cough", "runny nose",
        "minor pain", "back pain", "headache", "fatigue", "tired", "dizziness",
        "nausea", "vomiting", "diarrhea", "stomach ache",
    }
    if any(t in text for t in walkin_terms):
        if community and community.get("pct_without_family_doctor", 0) > 25:
            return CareDestination.upcc  # equity escalation in under-served areas
        return CareDestination.walk_in

    # --- Duration-based escalation to walk-in ---
    if duration and any(w in duration.lower() for w in ["week", "month", "long time"]):
        return CareDestination.walk_in

    return CareDestination.call_811


# ---------------------------------------------------------------------------
# Helper: wait times context string
# ---------------------------------------------------------------------------


def get_wait_times_context(wait_times_data: pd.DataFrame) -> Optional[str]:
    """
    Filter BC wait times to the most recent year, compute average median wait,
    and return a human-readable context string.
    """
    try:
        if wait_times_data.empty:
            return None

        bc_df = wait_times_data[wait_times_data["province"].str.upper() == "BC"].copy()
        if bc_df.empty:
            return None

        most_recent_year = bc_df["year"].max()
        recent_bc = bc_df[bc_df["year"] == most_recent_year]
        avg_wait = round(recent_bc["median_wait_days"].mean(), 1)

        return (
            f"BC's average surgical wait time is {avg_wait} days — "
            "getting to the right provider now helps prevent longer waits later."
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Helper: call Claude (wrapped for asyncio executor)
# ---------------------------------------------------------------------------


def _call_claude_sync(model: str, system: str, messages: list[dict], max_tokens: int = 1024) -> str:
    """Synchronous Anthropic SDK call. Run via run_in_executor."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
    )
    return response.content[0].text


async def _call_claude(model: str, system: str, messages: list[dict], max_tokens: int = 1024) -> str:
    """Async wrapper around synchronous Anthropic SDK."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: _call_claude_sync(model, system, messages, max_tokens),
    )


# ---------------------------------------------------------------------------
# Chat system prompt
# ---------------------------------------------------------------------------

CHAT_SYSTEM_PROMPT = """You are a friendly BC healthcare navigation assistant. Your job is to help people figure out where to go for care — not to diagnose them.

RULES:
- Speak at a grade 6 reading level. Be warm, calm, and reassuring.
- NEVER diagnose. NEVER say "you have X". NEVER recommend specific medications.
- Ask about: what is happening, how long it has been going on, how bad it feels (1-10), and where in the body.
- After 3-4 exchanges where you have gathered enough information, respond with ONLY a JSON object — nothing else.
- If you do not yet have enough information, respond with ONLY your next question as plain text — no JSON.

WHEN YOU HAVE ENOUGH INFO, respond with ONLY this JSON (no extra text):
{
  "ready": true,
  "symptoms": ["symptom1", "symptom2"],
  "duration": "e.g. 2 days",
  "severity": "e.g. moderate, 6/10",
  "plain_descriptions": [
    {"patient_words": "my tummy hurts", "clinical_term": "abdominal pain"},
    {"patient_words": "I feel dizzy", "clinical_term": "dizziness"}
  ]
}

IMPORTANT: Only output the JSON when you are confident you have: what symptoms, how long, how severe. Otherwise ask your next question."""


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
        message="BC Care Navigator API is running.",
    )


@app.post(
    "/api/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Conversational symptom collection",
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Accepts conversation history and returns the next assistant message.
    Detects emergencies immediately. When enough info is gathered, returns
    extracted symptoms ready for navigation.
    """
    # 1. Emergency keyword check across ALL user messages
    if _any_message_emergency(request.messages):
        return ChatResponse(
            reply=(
                "This sounds like a medical emergency. "
                "Please call 9-1-1 immediately or have someone take you to the nearest Emergency Room right away. "
                "Do not drive yourself. Stay on the line with 9-1-1 — they will help you."
            ),
            is_emergency=True,
            ready_for_review=False,
        )

    # 2. Build messages list for Claude
    claude_messages = [
        {"role": m.role, "content": m.content}
        for m in request.messages
    ]

    # 3. Call Claude claude-sonnet-4-6
    try:
        raw_reply = await _call_claude(
            model="claude-sonnet-4-6",
            system=CHAT_SYSTEM_PROMPT,
            messages=claude_messages,
            max_tokens=512,
        )
    except Exception as exc:
        print(f"[chat] Claude call failed: {exc}")
        return ChatResponse(
            reply=(
                "I want to make sure I understand what you're going through. "
                "Can you tell me a bit more — where in your body does it hurt, "
                "how long has this been going on, and how bad would you say it feels from 1 to 10?"
            ),
            ready_for_review=False,
            is_emergency=False,
        )

    # 4. Try to parse as JSON with "ready": true
    stripped = raw_reply.strip()
    # Extract JSON block if Claude wrapped it in markdown fences
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if json_match:
        stripped = json_match.group(1)

    extracted_symptoms = None
    ready = False

    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
            if parsed.get("ready") is True:
                ready = True
                extracted_symptoms = ExtractedSymptoms(
                    symptoms=parsed.get("symptoms", []),
                    duration=parsed.get("duration"),
                    severity=parsed.get("severity"),
                    plain_descriptions=parsed.get("plain_descriptions", []),
                )
                # Build a friendly bridging message
                raw_reply = (
                    "Thanks for sharing that with me. I have enough information now. "
                    "Let me find the best place for you to get care."
                )
        except (json.JSONDecodeError, Exception) as exc:
            print(f"[chat] JSON parse failed: {exc}")
            # Fall through — treat as plain text reply

    return ChatResponse(
        reply=raw_reply,
        ready_for_review=ready,
        extracted_symptoms=extracted_symptoms,
        is_emergency=False,
    )


@app.post(
    "/api/navigate",
    response_model=NavigationResponse,
    tags=["Navigation"],
    summary="Determine care destination and generate guidance",
)
async def navigate(request: NavigationRequest) -> NavigationResponse:
    """
    Takes extracted symptoms and location, applies the triage rule tree,
    then calls Claude Haiku to generate a patient-friendly navigation response.
    """
    symptoms = request.extracted_symptoms.symptoms
    severity = request.extracted_symptoms.severity
    duration = request.extracted_symptoms.duration
    location = request.location

    # 1. Emergency keyword check in symptom list
    symptom_text = " ".join(symptoms)
    if _contains_emergency_keyword(symptom_text):
        return NavigationResponse(
            destination=CareDestination.call_911,
            destination_label=DESTINATION_LABELS[CareDestination.call_911],
            is_emergency=True,
            headline="Call 9-1-1 Now",
            reasoning=(
                "Your symptoms may be life-threatening. "
                "Emergency services can provide the fastest and safest care."
            ),
            what_to_bring=["Your health card (if available)", "A list of any medications you take"],
            what_to_say=["I am having a medical emergency", "My symptoms are: " + symptom_text],
            community_note=None,
            community_context=None,
            safety_triggered=True,
            wait_times_context=None,
        )

    # 2. Lookup community data
    bc_data = app.state.bc_data
    community = lookup_community(location, bc_data)

    # 3. Rule-based destination
    destination = determine_destination(symptoms, severity, duration, community)

    # 4. Wait times context
    wait_times_context = get_wait_times_context(app.state.wait_times_data)

    # 5. Build community context model (if matched)
    community_context_model = None
    if community:
        try:
            community_context_model = CommunityContext(
                chsa_name=str(community.get("chsa_name", "")),
                health_authority=str(community.get("health_authority", "")),
                pct_without_family_doctor=float(community.get("pct_without_family_doctor", 0)),
                er_visits_per_1000=float(community.get("er_visits_per_1000", 0)),
                opioid_overdose_rate=float(community.get("opioid_overdose_rate", 0)),
                pct_below_poverty_line=float(community.get("pct_below_poverty_line", 0)),
            )
        except Exception as exc:
            print(f"[navigate] Could not build CommunityContext: {exc}")

    # 6. Call Claude Haiku for patient-friendly guidance
    destination_label = DESTINATION_LABELS.get(destination, str(destination))
    community_info = ""
    if community:
        community_info = (
            f"\nCommunity context: {community.get('chsa_name', 'unknown')} "
            f"({community.get('health_authority', '')}). "
            f"{community.get('pct_without_family_doctor', 'N/A')}% without a family doctor. "
            f"ER visits: {community.get('er_visits_per_1000', 'N/A')} per 1,000 residents."
        )

    haiku_system = (
        "You are a BC healthcare navigation assistant. "
        "You have already decided where the patient should go. "
        "Your job is to write warm, clear, grade-6-level guidance to help them. "
        "Return ONLY a valid JSON object with these exact keys: "
        "headline (short, action-oriented, max 10 words), "
        "reasoning (2-3 sentences explaining why this destination), "
        "what_to_bring (list of 3-5 strings), "
        "what_to_say (list of 2-4 strings — exact phrases for the patient to use), "
        "community_note (1 sentence about local context, or null). "
        "Do not include any text outside the JSON."
    )

    haiku_user = (
        f"Patient symptoms: {', '.join(symptoms)}\n"
        f"Duration: {duration or 'unknown'}\n"
        f"Severity: {severity or 'unknown'}\n"
        f"Decided destination: {destination_label}\n"
        f"Location: {location}{community_info}\n\n"
        "Generate the patient guidance JSON now."
    )

    # Sensible defaults in case Claude fails
    default_headline = f"Go to: {destination_label}"
    default_reasoning = (
        f"Based on your symptoms, {destination_label} is the most appropriate level of care for you right now."
    )
    default_what_to_bring = [
        "Your BC Services Card or health card",
        "A list of any medications you are taking",
        "A trusted friend or family member if possible",
    ]
    default_what_to_say = [
        f"I was advised to come here by BC Care Navigator",
        f"My main symptoms are: {', '.join(symptoms[:3])}",
        f"This has been going on for: {duration or 'some time'}",
    ]

    headline = default_headline
    reasoning = default_reasoning
    what_to_bring = default_what_to_bring
    what_to_say = default_what_to_say
    community_note = None

    try:
        haiku_reply = await _call_claude(
            model="claude-haiku-4-5-20251001",
            system=haiku_system,
            messages=[{"role": "user", "content": haiku_user}],
            max_tokens=600,
        )

        # Strip markdown fences if present
        haiku_stripped = haiku_reply.strip()
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", haiku_stripped, re.DOTALL)
        if fence_match:
            haiku_stripped = fence_match.group(1)

        haiku_json = json.loads(haiku_stripped)
        headline = haiku_json.get("headline", default_headline)
        reasoning = haiku_json.get("reasoning", default_reasoning)
        raw_bring = haiku_json.get("what_to_bring", default_what_to_bring)
        raw_say = haiku_json.get("what_to_say", default_what_to_say)
        community_note = haiku_json.get("community_note") or None

        # Ensure lists of strings
        what_to_bring = [str(item) for item in raw_bring] if isinstance(raw_bring, list) else default_what_to_bring
        what_to_say = [str(item) for item in raw_say] if isinstance(raw_say, list) else default_what_to_say

    except Exception as exc:
        print(f"[navigate] Claude Haiku call failed or parse error: {exc}. Using defaults.")

    return NavigationResponse(
        destination=destination,
        destination_label=destination_label,
        is_emergency=False,
        headline=headline,
        reasoning=reasoning,
        what_to_bring=what_to_bring,
        what_to_say=what_to_say,
        community_note=community_note,
        community_context=community_context_model,
        safety_triggered=False,
        wait_times_context=wait_times_context,
    )


# ---------------------------------------------------------------------------
# Static files — mount frontend AFTER all API routes
# ---------------------------------------------------------------------------

_frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
else:
    print(f"[startup] WARNING: frontend directory not found at {_frontend_dir}")


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    PORT = int(os.getenv("PORT", "8000"))
    DEBUG = os.getenv("DEBUG", "true").lower() == "true"
    uvicorn.run("backend.main:app", host="0.0.0.0", port=PORT, reload=DEBUG)
