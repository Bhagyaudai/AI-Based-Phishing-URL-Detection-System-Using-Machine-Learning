"""Initial prediction engine for phishing URL risk scoring."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import joblib
import pandas as pd

from .evaluation import positive_class_scores
from .features import FEATURE_COLUMNS, explain_feature_indicators, extract_url_features


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionResult:
    """Structured output returned by the URL prediction engine."""

    risk_score: int
    phishing_probability: float
    confidence_score: int
    threat_level: str
    reasoning: List[str]
    recommended_action: str


class HeuristicFallbackModel:
    """Local scorer used when no trained model artifact is available yet."""

    classes_ = [0, 1]

    def predict_proba(self, features: pd.DataFrame) -> list[list[float]]:
        probabilities: list[list[float]] = []
        for _, row in features.iterrows():
            phishing_probability = self._row_probability(row.to_dict())
            probabilities.append([1.0 - phishing_probability, phishing_probability])
        return probabilities

    def predict(self, features: pd.DataFrame) -> list[int]:
        return [1 if row[1] >= 0.5 else 0 for row in self.predict_proba(features)]

    @staticmethod
    def _row_probability(features: Mapping[str, float]) -> float:
        structural_points = heuristic_risk_points(features)
        probability = structural_points / 82.0
        if features.get("brand_spoofing_score", 0) >= 0.8:
            probability += 0.12
        if features.get("typosquatting_score", 0) >= 0.78:
            probability += 0.12
        if features.get("homograph_score", 0) >= 0.7:
            probability += 0.1
        return max(0.05, min(0.95, probability))


def heuristic_risk_points(features: Mapping[str, float]) -> int:
    """Calculate supplemental risk points from high-signal URL indicators."""

    points = 0
    if features.get("has_ip_address", 0) >= 1:
        points += 18
    if features.get("uses_https", 0) == 0:
        points += 8
    if features.get("ssl_valid", -1) == 0:
        points += 12
    if features.get("suspicious_word_count", 0) >= 2:
        points += 14
    elif features.get("suspicious_word_count", 0) == 1:
        points += 7
    if features.get("encoded_char_count", 0) >= 2:
        points += 10
    if features.get("has_embedded_url", 0) >= 1:
        points += 15
    if features.get("suspicious_query_key_count", 0) >= 1:
        points += 8
    if features.get("subdomain_count", 0) >= 3:
        points += 8
    if features.get("has_punycode", 0) >= 1:
        points += 10
    if features.get("domain_age_days", 999999) != -1 and features.get("domain_age_days", 999999) < 30:
        points += 12
    if features.get("redirect_count", 0) >= 2:
        points += 10
    if features.get("shortener_hint", 0) >= 1:
        points += 6
    if features.get("url_length", 0) >= 120:
        points += 6
    if features.get("url_entropy", 0) >= 4.5:
        points += 6
    if features.get("brand_spoofing_score", 0) >= 0.8:
        points += 18
    if features.get("typosquatting_score", 0) >= 0.78:
        points += 18
    elif features.get("typosquatting_score", 0) >= 0.6:
        points += 10
    if features.get("domain_impersonation_score", 0) >= 0.7:
        points += 16
    if features.get("homograph_score", 0) >= 0.7:
        points += 16
    if features.get("url_mutation_score", 0) >= 0.45:
        points += 10
    if features.get("char_pattern_risk", 0) >= 0.45:
        points += 8
    if features.get("domain_reputation_risk", 0) >= 0.45:
        points += 12
    if features.get("max_brand_similarity", 0) >= 0.88 and features.get("brand_spoofing_score", 0) > 0:
        points += 8
    return min(points, 70)


def strong_deceptive_evidence(features: Mapping[str, float]) -> int:
    """Count severe indicators that override trusted-domain reassurance."""

    checks = [
        features.get("has_ip_address", 0) >= 1,
        features.get("has_punycode", 0) >= 1,
        features.get("brand_spoofing_score", 0) >= 0.8,
        features.get("typosquatting_score", 0) >= 0.78,
        features.get("domain_impersonation_score", 0) >= 0.7,
        features.get("homograph_score", 0) >= 0.7,
        features.get("has_embedded_url", 0) >= 1,
        features.get("redirect_count", 0) >= 3,
        features.get("domain_reputation_risk", 0) >= 0.45,
    ]
    return sum(bool(check) for check in checks)


def trusted_domain_risk_adjustment(
    probability: float,
    risk_score: int,
    features: Mapping[str, float],
) -> tuple[float, int, list[str]]:
    """Reduce false positives for known domains when URL evidence is normal."""

    if features.get("trusted_domain_match", 0) < 1:
        return probability, risk_score, []

    normality = features.get("trusted_domain_normality_score", 0)
    severe_count = strong_deceptive_evidence(features)
    if normality < 0.78 or severe_count > 0:
        return probability, risk_score, []

    adjusted_probability = min(probability, 0.18)
    adjusted_risk = min(risk_score, 18)
    return adjusted_probability, adjusted_risk, [
        "Known trusted domain with normal security and URL structure"
    ]


def evidence_count(features: Mapping[str, float]) -> int:
    """Count high-signal indicators supporting the risk decision."""

    checks = [
        features.get("has_ip_address", 0) >= 1,
        features.get("uses_https", 0) == 0,
        features.get("ssl_valid", -1) == 0,
        features.get("suspicious_word_count", 0) >= 1,
        features.get("encoded_char_count", 0) >= 2,
        features.get("has_embedded_url", 0) >= 1,
        features.get("suspicious_query_key_count", 0) >= 1,
        features.get("subdomain_count", 0) >= 3,
        features.get("has_punycode", 0) >= 1,
        features.get("domain_age_days", 999999) != -1 and features.get("domain_age_days", 999999) < 30,
        features.get("redirect_count", 0) >= 2,
        features.get("shortener_hint", 0) >= 1,
        features.get("url_entropy", 0) >= 4.5,
        features.get("brand_spoofing_score", 0) >= 0.8,
        features.get("typosquatting_score", 0) >= 0.78,
        features.get("domain_impersonation_score", 0) >= 0.7,
        features.get("homograph_score", 0) >= 0.7,
        features.get("url_mutation_score", 0) >= 0.45,
        features.get("char_pattern_risk", 0) >= 0.45,
        features.get("domain_reputation_risk", 0) >= 0.45,
        features.get("trusted_domain_normality_score", 0) >= 0.8,
    ]
    return sum(bool(check) for check in checks)


def confidence_from_evidence(probability: float, risk_score: int, features: Mapping[str, float]) -> int:
    """Estimate confidence using model certainty and independent evidence depth."""

    model_certainty = abs(probability - 0.5) * 100
    evidence_depth = min(35, evidence_count(features) * 5)
    risk_alignment = 15 if (probability >= 0.65 and risk_score >= 60) or (probability <= 0.25 and risk_score < 40) else 0
    confidence = 45 + model_certainty * 0.45 + evidence_depth + risk_alignment
    return int(round(max(0, min(100, confidence))))


def threat_level_from_score(risk_score: int) -> str:
    """Map a numeric risk score to the dissertation threat levels."""

    if risk_score >= 80:
        return "Critical"
    if risk_score >= 60:
        return "High Risk"
    if risk_score >= 40:
        return "Medium Risk"
    if risk_score >= 20:
        return "Low Risk"
    return "Safe"


def recommended_action(threat_level: str) -> str:
    """Return the action guidance associated with a threat level."""

    actions = {
        "Safe": "Allow with normal monitoring",
        "Low Risk": "Allow, but log the URL if the business context is sensitive",
        "Medium Risk": "Warn the user and request verification before access",
        "High Risk": "Block Access and queue for security review",
        "Critical": "Block Access immediately and escalate for security review",
    }
    return actions[threat_level]


class PhishingPredictionEngine:
    """Load a trained classifier and score URLs with explanation."""

    def __init__(
        self,
        model: Any,
        feature_columns: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.feature_columns = feature_columns or FEATURE_COLUMNS

    @classmethod
    def from_files(
        cls,
        model_path: str | Path,
        metadata_path: Optional[str | Path] = None,
    ) -> "PhishingPredictionEngine":
        """Create an engine from saved model and metadata files."""

        LOGGER.info("Loading phishing detection model from %s", model_path)
        model = joblib.load(model_path)
        feature_columns = FEATURE_COLUMNS
        if metadata_path is not None:
            try:
                metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
                feature_columns = metadata.get("feature_columns", FEATURE_COLUMNS)
            except (OSError, json.JSONDecodeError) as exc:
                LOGGER.warning("Could not read model metadata from %s: %s", metadata_path, exc)
        return cls(model=model, feature_columns=feature_columns)

    def build_feature_frame(
        self,
        url: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> tuple[pd.DataFrame, Dict[str, float]]:
        """Extract and align features for one URL."""

        feature_values = extract_url_features(url, metadata)
        frame = pd.DataFrame([feature_values])
        frame = frame.reindex(columns=self.feature_columns, fill_value=0.0)
        return frame, feature_values

    def model_probability(self, feature_frame: pd.DataFrame) -> float:
        """Return model phishing probability with defensive error handling."""

        try:
            probability = float(positive_class_scores(self.model, feature_frame)[0])
        except Exception as exc:  # pragma: no cover - defensive production guard
            LOGGER.exception("Model probability calculation failed: %s", exc)
            return 0.5
        return max(0.0, min(1.0, probability))

    def predict(
        self,
        url: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> PredictionResult:
        """Score a URL and return structured risk output."""

        try:
            feature_frame, feature_values = self.build_feature_frame(url, metadata)
        except Exception as exc:
            LOGGER.exception("Feature extraction failed for URL %r: %s", url, exc)
            return PredictionResult(
                risk_score=100,
                phishing_probability=1.0,
                confidence_score=90,
                threat_level="Critical",
                reasoning=["URL could not be parsed safely by the detection engine"],
                recommended_action=recommended_action("Critical"),
            )

        probability = self.model_probability(feature_frame)
        heuristic_points = heuristic_risk_points(feature_values)
        risk_score = int(round(min(100.0, probability * 68.0 + heuristic_points)))
        probability, risk_score, trust_reasons = trusted_domain_risk_adjustment(
            probability,
            risk_score,
            feature_values,
        )
        threat_level = threat_level_from_score(risk_score)
        reasoning = explain_feature_indicators(feature_values)
        for reason in reversed(trust_reasons):
            reasoning.insert(0, reason)
        confidence_score = confidence_from_evidence(probability, risk_score, feature_values)

        if probability >= 0.85 and "Model probability strongly indicates phishing risk" not in reasoning:
            reasoning.insert(0, "Model probability strongly indicates phishing risk")
        elif probability <= 0.20:
            reasoning.insert(0, "Model probability indicates low phishing likelihood")

        LOGGER.info(
            "URL scored: threat_level=%s risk_score=%s confidence_score=%s",
            threat_level,
            risk_score,
            confidence_score,
        )

        return PredictionResult(
            risk_score=risk_score,
            phishing_probability=round(probability, 6),
            confidence_score=confidence_score,
            threat_level=threat_level,
            reasoning=reasoning,
            recommended_action=recommended_action(threat_level),
        )


def main() -> None:
    """Command-line entry point for single URL scoring."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Score one URL for phishing risk")
    parser.add_argument("--model", required=True, help="Path to best_model.joblib")
    parser.add_argument("--metadata", help="Path to training_metadata.json")
    parser.add_argument("--url", required=True, help="URL to score")
    args = parser.parse_args()

    engine = PhishingPredictionEngine.from_files(args.model, args.metadata)
    result = engine.predict(args.url)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
