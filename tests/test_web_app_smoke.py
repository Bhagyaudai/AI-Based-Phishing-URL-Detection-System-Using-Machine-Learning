from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB_APP = ROOT / "web_app"
if str(WEB_APP) not in sys.path:
    sys.path.insert(0, str(WEB_APP))

import services


class MemoryUpload:
    def __init__(self, filename: str, content: str) -> None:
        self.filename = filename
        self._content = content.encode("utf-8")

    def save(self, destination) -> None:
        Path(destination).write_bytes(self._content)


def configure_service_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(services, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(services, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(services, "STORAGE_DIR", tmp_path / "storage")
    monkeypatch.setattr(services, "HISTORY_PATH", tmp_path / "storage" / "history.json")
    monkeypatch.setattr(services, "EVENT_LOG_PATH", tmp_path / "storage" / "events.json")
    monkeypatch.setattr(services, "TRAINING_STATUS_PATH", tmp_path / "storage" / "training_status.json")
    monkeypatch.setattr(services, "MODEL_PATH", tmp_path / "models" / "best_model.joblib")
    monkeypatch.setattr(services, "METADATA_PATH", tmp_path / "models" / "training_metadata.json")


def test_detection_service_scores_and_stores_history(tmp_path, monkeypatch):
    configure_service_paths(tmp_path, monkeypatch)
    service = services.DetectionService()
    result = service.analyze_url("http://g00gle-login.com")

    assert result["threat_level"] in {"High Risk", "Critical"}
    assert result["risk_score"] >= 60
    assert 0 <= result["confidence_score"] <= 100
    assert result["model_source"] == "heuristic_fallback"
    assert service.get_history()[0]["url"] == "http://g00gle-login.com"
    assert service.get_logs()[0]["event_type"] == "URL_SCAN"


def test_training_failure_is_recorded_with_reason(tmp_path, monkeypatch):
    configure_service_paths(tmp_path, monkeypatch)
    service = services.DetectionService()
    upload = MemoryUpload("bad.csv", "url,label\nhttps://www.google.com/,0\nhttp://g00gle-login.com,1\n")

    try:
        service.train_from_upload(upload)
    except services.ServiceError as exc:
        assert "at least two legitimate and two phishing" in str(exc)
    else:
        raise AssertionError("Training should fail for an undersized dataset")

    assert service.get_training_status()["status"] == "FAILED"
    assert service.get_logs()[0]["event_type"] == "TRAINING_FAILURE"
