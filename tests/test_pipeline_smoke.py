from __future__ import annotations

import pandas as pd

from phishing_detection.features import extract_url_features
from phishing_detection.prediction_engine import PhishingPredictionEngine
from phishing_detection.preprocessing import preprocess_dataset


class FixedRiskModel:
    def predict_proba(self, features):
        return [[0.08, 0.92] for _ in range(len(features))]

    def predict(self, features):
        return [1 for _ in range(len(features))]


class FixedLowModel:
    def predict_proba(self, features):
        return [[0.80, 0.20] for _ in range(len(features))]

    def predict(self, features):
        return [0 for _ in range(len(features))]


def test_feature_extraction_flags_obvious_risk_indicators():
    features = extract_url_features(
        "http://192.0.2.10/login/verify?redirect=https%3A%2F%2Ffake.example",
        {"ssl_valid": 0, "redirect_count": 3, "domain_age_days": 5},
    )

    assert features["has_ip_address"] == 1
    assert features["uses_https"] == 0
    assert features["suspicious_word_count"] >= 2
    assert features["has_embedded_url"] == 1
    assert features["redirect_count"] == 3
    assert "url_mutation_score" in features


def test_preprocessing_returns_clean_feature_matrix():
    raw = pd.DataFrame(
        {
            "url": [
                "https://www.bits-pilani.ac.in/",
                "http://secure-login-example.com/verify",
                "https://www.wipro.com/",
                "http://secure-login-example.com/verify",
                "http://192.0.2.10/account/update",
            ],
            "label": [0, 1, "legitimate", "phishing", 1],
            "source": ["allowlist", "feed", "allowlist", "feed", "feed"],
            "ssl_valid": [1, 0, 1, 0, 0],
        }
    )

    dataset = preprocess_dataset(raw)

    assert len(dataset.cleaned_records) == 4
    assert dataset.quality_report["duplicate_urls_removed"] == 1
    assert dataset.quality_report["class_distribution"] == {"legitimate": 2, "phishing": 2}
    assert "url_length" in dataset.features.columns


def test_prediction_engine_returns_required_fields():
    engine = PhishingPredictionEngine(FixedRiskModel())
    result = engine.predict("http://secure-login-example.com/account/verify")

    assert result.risk_score >= 70
    assert result.threat_level in {"High Risk", "Critical"}
    assert result.phishing_probability == 0.92
    assert 0 <= result.confidence_score <= 100
    assert result.reasoning
    assert "recommended" not in result.recommended_action.lower()


def test_advanced_detection_flags_unseen_brand_spoofing_urls():
    engine = PhishingPredictionEngine(FixedLowModel())

    google_result = engine.predict("http://google-login-secure.com")
    leetspeak_result = engine.predict("http://g00gle-login.com")
    amazon_result = engine.predict("http://amazon-secure-authentication.com")

    assert google_result.threat_level in {"High Risk", "Critical"}
    assert leetspeak_result.threat_level in {"High Risk", "Critical"}
    assert amazon_result.risk_score >= 55
    assert any("trusted brand" in reason.lower() for reason in google_result.reasoning)
    assert any("typosquatting" in reason.lower() for reason in leetspeak_result.reasoning)


def test_trusted_domain_with_normal_structure_stays_safe():
    engine = PhishingPredictionEngine(FixedRiskModel())
    result = engine.predict(
        "https://www.google.com/search?q=security",
        {"ssl_valid": 1, "domain_age_days": 9000, "reputation_score": 98, "redirect_count": 0},
    )

    assert result.threat_level == "Safe"
    assert result.risk_score <= 18
    assert any("known trusted domain" in reason.lower() for reason in result.reasoning)


def test_microsoft_service_links_with_long_ids_stay_safe():
    engine = PhishingPredictionEngine(FixedRiskModel())
    urls = [
        "https://outlook.cloud.microsoft/mail/inbox/id/AAQkADAxOTgzN2VhLTY0ODMtwJmJgOZgA",
        "https://copilot.microsoft.com/chats/AAQkADAxOTgzN2VhLTY0ODMtwJmJgOZgA",
    ]

    for url in urls:
        result = engine.predict(
            url,
            {"ssl_valid": 1, "domain_age_days": 9000, "reputation_score": 98, "redirect_count": 0},
        )
        assert result.threat_level == "Safe"
        assert result.risk_score <= 18
