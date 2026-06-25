"""Application services for model scoring, retraining, and local history."""

from __future__ import annotations

import json
import logging
import re
import shutil
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phishing_detection.features import FEATURE_COLUMNS
from phishing_detection.models import train_from_csv
from phishing_detection.preprocessing import encode_label
from phishing_detection.prediction_engine import HeuristicFallbackModel, PhishingPredictionEngine


LOGGER = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = APP_DIR / "artifacts"
UPLOAD_DIR = ARTIFACT_DIR / "uploads"
MODEL_DIR = ARTIFACT_DIR / "models"
STORAGE_DIR = APP_DIR / "storage"
HISTORY_PATH = STORAGE_DIR / "history.json"
EVENT_LOG_PATH = STORAGE_DIR / "events.json"
TRAINING_STATUS_PATH = STORAGE_DIR / "training_status.json"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
METADATA_PATH = MODEL_DIR / "training_metadata.json"
MAX_BATCH_ROWS = 500


class ServiceError(ValueError):
    """Raised for user-correctable web application errors."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_directories() -> None:
    for path in [UPLOAD_DIR, MODEL_DIR, STORAGE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def clean_filename(filename: str) -> str:
    """Create a safe local filename for uploaded CSV files."""

    base = Path(filename or "dataset.csv").name
    base = re.sub(r"[^a-zA-Z0-9._-]+", "_", base).strip("._")
    if not base.lower().endswith(".csv"):
        base = f"{base}.csv"
    return base or "dataset.csv"


def validate_csv_upload(uploaded_file: Any) -> None:
    if uploaded_file is None or not getattr(uploaded_file, "filename", ""):
        raise ServiceError("Please upload a CSV file.")
    if not uploaded_file.filename.lower().endswith(".csv"):
        raise ServiceError("Only CSV files are supported.")


def summarize_training(metadata: Dict[str, Any]) -> Dict[str, Any]:
    quality = metadata.get("quality_report", {})
    best_name = metadata.get("best_model_name", "Unknown model")
    best_result = next(
        (
            result
            for result in metadata.get("model_results", [])
            if result.get("model_name") == best_name
        ),
        {},
    )
    return {
        "status": "SUCCESS",
        "best_model_name": best_name,
        "class_distribution": quality.get("class_distribution", {}),
        "cleaned_records": quality.get("cleaned_records"),
        "duplicate_urls_removed": quality.get("duplicate_urls_removed"),
        "test_metrics": best_result.get("test_metrics", {}),
        "feature_count": len(metadata.get("feature_columns", FEATURE_COLUMNS)),
        "artifact_saved": True,
    }


def validate_training_frame(frame: pd.DataFrame) -> Dict[str, Any]:
    """Validate a training CSV before any model artifacts are replaced."""

    missing_columns = [column for column in ["url", "label"] if column not in frame.columns]
    if missing_columns:
        raise ServiceError("Training CSV must contain 'url' and 'label' columns.")
    if frame.empty:
        raise ServiceError("Training CSV is empty.")

    labels = frame["label"].map(encode_label)
    distribution = {
        "legitimate": int((labels == 0).sum()),
        "phishing": int((labels == 1).sum()),
    }
    if distribution["legitimate"] < 2 or distribution["phishing"] < 2:
        raise ServiceError("Training requires at least two legitimate and two phishing records.")

    imbalance_ratio = max(distribution.values()) / max(1, min(distribution.values()))
    return {
        "records": int(len(frame)),
        "class_distribution": distribution,
        "imbalance_ratio": round(float(imbalance_ratio), 3),
    }


class DetectionService:
    """Coordinates model artifacts, URL scoring, CSV training, and history."""

    def __init__(self) -> None:
        ensure_directories()
        self.engine = self._load_engine()

    @property
    def has_trained_model(self) -> bool:
        return MODEL_PATH.exists()

    def _load_engine(self) -> PhishingPredictionEngine:
        if MODEL_PATH.exists():
            LOGGER.info("Using trained phishing detection model at %s", MODEL_PATH)
            return PhishingPredictionEngine.from_files(MODEL_PATH, METADATA_PATH if METADATA_PATH.exists() else None)

        LOGGER.info("No trained model artifact found; using local heuristic scorer")
        return PhishingPredictionEngine(HeuristicFallbackModel(), FEATURE_COLUMNS)

    def reload_engine(self) -> None:
        self.engine = self._load_engine()

    def analyze_url(
        self,
        url: str,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "web",
        store_history: bool = True,
    ) -> Dict[str, Any]:
        url = (url or "").strip()
        if not url:
            raise ServiceError("Enter a URL to analyze.")

        result = asdict(self.engine.predict(url, metadata))
        response = {
            "url": url,
            "analyzed_at": utc_now(),
            "model_source": "trained_model" if self.has_trained_model else "heuristic_fallback",
            **result,
        }
        self.log_event(
            "EXTENSION_SCAN" if source == "chrome_extension" else "URL_SCAN",
            {
                "url": url,
                "source": source,
                "threat_level": response["threat_level"],
                "risk_score": response["risk_score"],
                "confidence_score": response["confidence_score"],
                "model_source": response["model_source"],
            },
        )
        if store_history:
            self._append_history({**response, "source": source})
        return response

    def train_from_upload(self, uploaded_file: Any) -> Dict[str, Any]:
        validate_csv_upload(uploaded_file)
        destination = self._save_upload(uploaded_file)
        self.log_event("CSV_UPLOAD", {"dataset_file": destination.name, "purpose": "training"})
        self._write_training_status(
            {
                "status": "TRAINING_STARTED",
                "message": "Training started",
                "dataset_file": destination.name,
                "started_at": utc_now(),
            }
        )
        self.log_event("TRAINING_STARTED", {"dataset_file": destination.name})

        try:
            frame = pd.read_csv(destination)
            validation_summary = validate_training_frame(frame)
            LOGGER.info("Training phishing model from uploaded dataset %s", destination)
            with tempfile.TemporaryDirectory(prefix="phishing_model_") as temporary_dir:
                metadata = train_from_csv(destination, temporary_dir)
                temporary_model = Path(temporary_dir) / "best_model.joblib"
                temporary_metadata = Path(temporary_dir) / "training_metadata.json"
                if not temporary_model.exists() or not temporary_metadata.exists():
                    raise ServiceError("Training completed without saving model artifacts.")
                shutil.copy2(temporary_model, MODEL_PATH)
                shutil.copy2(temporary_metadata, METADATA_PATH)

            self.reload_engine()
            summary = summarize_training(metadata)
            summary["message"] = "Training successful"
            summary["trained_at"] = utc_now()
            summary["dataset_file"] = destination.name
            summary["validation"] = validation_summary
            summary["artifact_paths"] = {
                "model": str(MODEL_PATH),
                "metadata": str(METADATA_PATH),
            }
            self._write_training_status(summary)
            self.log_event("TRAINING_SUCCESS", summary)
            return summary
        except Exception as exc:
            failure = {
                "status": "FAILED",
                "message": "Training failed",
                "dataset_file": destination.name,
                "failed_at": utc_now(),
                "reason": str(exc),
            }
            self._write_training_status(failure)
            self.log_event("TRAINING_FAILURE", failure)
            LOGGER.exception("Training failed for uploaded dataset %s: %s", destination, exc)
            if isinstance(exc, ServiceError):
                raise
            raise ServiceError(str(exc)) from exc

    def batch_analyze(self, uploaded_file: Any) -> Dict[str, Any]:
        validate_csv_upload(uploaded_file)
        destination = self._save_upload(uploaded_file)
        self.log_event("CSV_UPLOAD", {"dataset_file": destination.name, "purpose": "batch_analysis"})
        frame = pd.read_csv(destination)
        if "url" not in frame.columns:
            raise ServiceError("Batch analysis CSV must contain a 'url' column.")
        if len(frame) > MAX_BATCH_ROWS:
            raise ServiceError(f"Batch analysis is limited to {MAX_BATCH_ROWS} URLs per upload.")

        records = []
        for _, row in frame.iterrows():
            metadata = {
                key: value
                for key, value in row.to_dict().items()
                if key not in {"url", "label"} and pd.notna(value)
            }
            records.append(
                self.analyze_url(
                    str(row["url"]),
                    metadata=metadata,
                    source="batch_upload",
                    store_history=False,
                )
            )

        self._append_history(
            {
                "url": f"Batch upload: {destination.name}",
                "analyzed_at": utc_now(),
                "source": "batch_upload",
                "risk_score": max((record["risk_score"] for record in records), default=0),
                "confidence_score": max((record["confidence_score"] for record in records), default=0),
                "threat_level": self._batch_threat_level(records),
                "reasoning": [f"Analyzed {len(records)} URLs from uploaded CSV"],
                "recommended_action": "Review high-risk and critical rows before allowing access",
            }
        )
        self.log_event(
            "BATCH_ANALYSIS",
            {
                "dataset_file": destination.name,
                "record_count": len(records),
                "highest_threat_level": self._batch_threat_level(records),
            },
        )
        return {
            "dataset_file": destination.name,
            "record_count": len(records),
            "results": records,
        }

    def get_history(self) -> list[Dict[str, Any]]:
        if not HISTORY_PATH.exists():
            return []
        try:
            return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOGGER.warning("History file is not valid JSON; resetting local history")
            return []

    def clear_history(self) -> None:
        HISTORY_PATH.write_text("[]", encoding="utf-8")
        self.log_event("HISTORY_CLEARED", {})

    def get_logs(self) -> list[Dict[str, Any]]:
        return self._read_json_list(EVENT_LOG_PATH)

    def clear_logs(self) -> None:
        EVENT_LOG_PATH.write_text("[]", encoding="utf-8")

    def get_training_status(self) -> Dict[str, Any]:
        if not TRAINING_STATUS_PATH.exists():
            return {
                "status": "NO_TRAINING_RUN",
                "message": "No model training has been run in this workspace.",
            }
        try:
            return json.loads(TRAINING_STATUS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"status": "UNKNOWN", "message": "Training status could not be read."}

    def model_status(self) -> Dict[str, Any]:
        metadata = {}
        if METADATA_PATH.exists():
            try:
                metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata = {}
        return {
            "model_loaded": self.has_trained_model,
            "model_source": "trained_model" if self.has_trained_model else "heuristic_fallback",
            "model_path": str(MODEL_PATH) if self.has_trained_model else None,
            "metadata_path": str(METADATA_PATH) if METADATA_PATH.exists() else None,
            "best_model_name": metadata.get("best_model_name"),
            "feature_count": len(metadata.get("feature_columns", FEATURE_COLUMNS)),
            "training_status": self.get_training_status(),
        }

    def _save_upload(self, uploaded_file: Any) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_{clean_filename(uploaded_file.filename)}"
        destination = UPLOAD_DIR / filename
        uploaded_file.save(destination)
        return destination

    def log_event(self, event_type: str, details: Dict[str, Any]) -> None:
        event = {
            "event_type": event_type,
            "logged_at": utc_now(),
            "details": details,
        }
        events = self.get_logs()
        events.insert(0, event)
        EVENT_LOG_PATH.write_text(json.dumps(events[:250], indent=2), encoding="utf-8")
        LOGGER.info("%s: %s", event_type, details)

    def _append_history(self, record: Dict[str, Any]) -> None:
        history = self.get_history()
        history.insert(0, record)
        HISTORY_PATH.write_text(json.dumps(history[:100], indent=2), encoding="utf-8")

    def _write_training_status(self, status: Dict[str, Any]) -> None:
        TRAINING_STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")

    @staticmethod
    def _read_json_list(path: Path) -> list[Dict[str, Any]]:
        if not path.exists():
            return []
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, list) else []
        except json.JSONDecodeError:
            return []

    @staticmethod
    def _batch_threat_level(records: Iterable[Dict[str, Any]]) -> str:
        levels = ["Safe", "Low Risk", "Medium Risk", "High Risk", "Critical"]
        highest = 0
        for record in records:
            try:
                highest = max(highest, levels.index(record.get("threat_level", "Safe")))
            except ValueError:
                continue
        return levels[highest]
