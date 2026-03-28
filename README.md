# Healthcare Hackathon Project

A health informatics project built for rapid prototyping with real patient data pipelines, FHIR integration, and ML-ready analysis.

---

## Problem

> **TODO:** Describe the clinical or operational problem you are solving.
>
> Example: _"Emergency departments lack real-time tools to triage patient risk, leading to delayed care for high-acuity patients."_

---

## Solution

> **TODO:** Describe your approach.
>
> Example: _"A lightweight risk-scoring API that ingests structured patient data (vitals, diagnoses, labs) and returns an acuity score with explainable features — deployable in any EHR workflow via FHIR R4."_

Key components:
- **Backend:** FastAPI REST API with a `/api/analyze` endpoint for patient data scoring
- **Data layer:** Pandas-based pipeline for EHR/CSV/FHIR data ingestion
- **ML:** scikit-learn models for risk stratification
- **Frontend:** Zero-dependency HTML/JS interface for demo purposes
- **Notebooks:** Jupyter EDA and model prototyping

---

## Setup

### Prerequisites
- Python 3.10+
- Node.js (optional, for frontend tooling)

### Backend

```bash
# Clone and enter the repo
git clone <your-repo-url>
cd healthcare-hackathon

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your values

# Run the API
uvicorn backend.main:app --reload --port 8000
```

API docs will be available at: http://localhost:8000/docs

### Frontend

Open `frontend/index.html` directly in a browser. The page will POST to `http://localhost:8000/api/analyze`.

### Notebooks

```bash
jupyter notebook notebooks/exploration.ipynb
```

### Data

Place raw data files in the `data/` directory. The directory is git-ignored for large files — do not commit PHI or patient data.

---

## Team

| Name | Role | Contact |
|------|------|---------|
| TODO | Lead Developer | |
| TODO | Data Scientist | |
| TODO | Clinician Advisor | |
| TODO | UX / Frontend | |

---

## Project Structure

```
healthcare-hackathon/
├── backend/
│   ├── main.py          # FastAPI application entrypoint
│   ├── models.py        # Pydantic request/response models
│   └── routes/          # Route modules (add new endpoints here)
├── data/                # Raw and processed data (git-ignored)
├── notebooks/
│   └── exploration.ipynb  # EDA and prototyping
├── frontend/
│   └── index.html       # Demo UI
├── .env.example         # Environment variable template
├── requirements.txt     # Python dependencies
└── README.md
```

---

## License

MIT — see LICENSE file. **Do not commit any PHI (Protected Health Information).**
