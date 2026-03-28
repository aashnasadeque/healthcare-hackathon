# BC Care Navigator

**UVic Healthcare AI Hackathon 2026 — Track 2: Population Health & Health Equity**

## The Problem
People in BC don't know where to go for care. They default to the ER — 8-10 hour waits — for conditions that could be handled at a walk-in clinic, UPCC, or pharmacy. In communities where 25%+ of residents have no family doctor, this problem is even worse.

## The Solution
A conversational care navigation tool. Patients describe their symptoms in plain language. The AI asks follow-up questions. Patients review a plain/clinical interpretation. Then they get one clear answer: where to go, why, what to bring, and what to say.

**Backed by BC community health data** — communities with fewer healthcare resources get recommendations that reflect their actual options.

## Team
- Aashna Sadeque

## Challenge Track
Track 2: Population Health & Health Equity

## Data Sources
- BC Community Health Indicators (78 BC communities)
- CIHI Wait Times (BC surgical wait time context)
- BC Opioid Surveillance Data

## Tech Stack
- **Backend**: Python, FastAPI
- **AI**: Claude API (Anthropic)
- **Data**: pandas, BC open health datasets
- **Frontend**: Vanilla HTML/CSS/JS (mobile-first)
- **Deployment**: Render

## Live Demo
**https://healthcare-hackathon-o6uw.onrender.com/**

## Problem Statement
BC residents without a family doctor have no clear pathway to specialist care. They default to emergency rooms for non-emergencies, contributing to 8-10 hour wait times. This tool uses BC population health data to route patients to the right level of care — accounting for their community's specific access gaps.

## How to Run Locally
```bash
git clone https://github.com/aashnasadeque/healthcare-hackathon
cd healthcare-hackathon
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
uvicorn backend.main:app --reload
# Open http://localhost:8000
```

## How Judging Criteria Are Met
- **Innovation**: Conversational triage with plain/clinical review screen — novel in BC healthcare navigation
- **Technical Execution**: FastAPI + Claude API + BC community data, live deployment
- **Impact Potential**: Directly reduces ER misuse, routes unattached patients to UPCCs that can write specialist referrals
- **Presentation**: Mobile-first, plain language, designed for real patients
- **Design/UX**: Single clear recommendation, color-coded, accessible
