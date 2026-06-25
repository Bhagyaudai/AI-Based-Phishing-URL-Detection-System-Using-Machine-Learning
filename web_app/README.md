# Phishing Detection Web App

This folder contains the local web application for URL analysis, CSV-based model
training, batch URL testing, and detection history. It uses the shared
`phishing_detection` machine-learning package from `src/`.

## Features

- Analyze one URL at a time.
- Upload a CSV dataset to retrain the phishing detection model.
- Upload a CSV file for batch URL testing.
- Show threat level, risk score, confidence score, explanation, and recommended
  action.
- Show training status, model status, operational logs, and recent detection
  history.
- Reuse the same local API used by the Chrome extension.

## CSV format

Training CSV files must include:

| Column | Required | Description |
| --- | --- | --- |
| `url` | yes | URL to learn from |
| `label` | yes | `1` for phishing, `0` for legitimate |

Optional metadata columns such as `domain_age_days`, `ssl_valid`,
`redirect_count`, `reputation_score`, `registrar`, and `whois_available` are
used when present.

Batch testing CSV files must include a `url` column. Optional metadata columns
are also used for scoring.

## Setup

From the repository root:

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .

## Run the web app

From the repository root:
cd .\web_app\
python app.py

Open:
http://127.0.0.1:5000

### Single URL analysis

1. Enter a suspicious or legitimate URL.
2. The app extracts lexical, domain, security, behavioral, similarity, and brand
   impersonation features.
3. The active model produces a phishing probability.
4. The prediction engine combines model probability with similarity and
   heuristic evidence.
5. The app returns threat level, risk score, confidence score, reasoning, and
   recommended action.

### CSV model training

1. Upload a CSV file with `url` and `label`.
2. The app records **Training started** and validates the CSV before replacing
   model artifacts.
3. The preprocessing pipeline validates records, removes duplicates, extracts
   features, and checks class balance.
4. Candidate models are trained and compared.
5. The best model is saved under `web_app/artifacts/models/` only after
   training succeeds and artifacts are verified.
6. The app shows **Training successful** with metrics, model name, class
   distribution, and artifact confirmation.
7. If validation or training fails, the app shows **Training failed** with the
   actual reason and keeps the previous model unchanged.

### CSV batch testing

1. Upload a CSV file with a `url` column.
2. Each URL is scored with the active detection engine.
3. The interface displays the first batch results with threat level, risk score,
   and confidence.

## Local files

Generated files are intentionally kept out of version control:

```text
web_app/artifacts/uploads/       Uploaded CSV files
web_app/artifacts/models/        Trained model artifacts
web_app/storage/history.json     Local detection history
web_app/storage/events.json      Local scan, upload, and training logs
web_app/storage/training_status.json  Latest training status
```

## Web app sections

- **URL Scan Page**: Analyze a single URL.
- **CSV Upload Page**: Upload URLs for batch testing.
- **Training Page**: Upload labeled CSV data and retrain the model.
- **Model Status Page**: View active model source, saved artifacts, feature
  count, and latest training status.
- **Result Explanation Page**: Review how risk score, confidence score, and
  trusted-domain handling are calculated.
- **Logs & History Page**: Review URL scans, extension scans, CSV uploads,
  training attempts, training successes, training failures, and errors.
