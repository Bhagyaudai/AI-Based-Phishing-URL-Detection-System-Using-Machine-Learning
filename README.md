# AI-Based Phishing URL Detection System Using Machine Learning

BITS Pilani WILP dissertation project for designing and implementing the
machine-learning core of an enterprise phishing URL detection system.

## Current milestone scope

This repository currently covers only the phases scheduled up to 30 June 2026:

1. Requirement analysis
2. Dataset collection strategy
3. Literature survey
4. Feature analysis
5. Dataset preprocessing
6. Feature extraction
7. Machine learning model development
8. Prediction system development

The current implementation includes the detection core, a local web application,
and a Chrome extension that calls the same local detection API. It does not
include dashboards, performance-optimization dashboards, cloud deployment,
authentication, or database integration.

## Repository structure

```text
.
├── data/
│   ├── processed/                 # Cleaned datasets produced by preprocessing
│   ├── raw/                       # Source CSV files from selected data providers
│   └── README.md                  # Dataset layout and handling notes
├── chrome_extension/              # Chrome extension for current-tab URL checks
├── docs/
│   ├── 01_requirements_analysis.md
│   ├── 02_literature_survey.md
│   ├── 03_dataset_architecture.md
│   ├── 04_feature_engineering.md
│   ├── 05_architecture_diagrams.md
│   ├── mid_sem_submission_report.md
│   └── mid_sem_presentation_outline.md
├── models/                        # Serialized trained models and metadata
├── src/phishing_detection/
│   ├── evaluation.py              # Model evaluation helpers
│   ├── features.py                # URL/domain/security/behavioral extraction
│   ├── models.py                  # Training and model comparison pipeline
│   ├── prediction_engine.py       # Initial URL risk scoring engine
│   └── preprocessing.py           # Production-style dataset preprocessing
├── web_app/                       # Local Flask app and CSV upload workflow
└── tests/
    └── test_pipeline_smoke.py     # Lightweight validation of core modules
```

## Quick start

Create a Python environment and install the required libraries:

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .

## Run the web app

From the repository root:
cd .\web_app\
python app.py

Open:
http://127.0.0.1:5000

Prepare a CSV using the schema described in `docs/03_dataset_architecture.md`.
The minimum required columns are:

- `url`
- `label` where phishing is `1` and legitimate is `0`

Train and evaluate the baseline models

Run a local prediction against a single URL after a model has been trained

The prediction result includes `risk_score`, `phishing_probability`,
`confidence_score`, `threat_level`, `reasoning`, and `recommended_action`.
The engine now includes brand similarity, typosquatting, homograph, URL
mutation, redirect, shortener, and domain reputation signals for URLs that may
not already exist in the training dataset. Known legitimate domains are handled
through trusted-domain structure and reputation checks so normal links remain
safe unless strong deceptive indicators are present.

The web app supports:

- Single URL analysis
- CSV upload for model retraining
- CSV upload for batch URL testing
- Local detection history
- Training status, model status, and operational logs
- Risk score, confidence score, threat level, reasoning, and recommended action

See `web_app/README.md` for detailed CSV and run instructions.

## Load the Chrome extension

1. Start the web app locally.
2. Open Chrome and go to `chrome://extensions`.
3. Enable Developer mode.
4. Click **Load unpacked**.
5. Select the `chrome_extension/` folder.

The extension analyzes the active tab by calling the local web app API. See
`chrome_extension/README.md` for the full loading and usage steps.
