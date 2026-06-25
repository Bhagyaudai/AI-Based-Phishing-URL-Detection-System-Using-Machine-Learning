"""Dataset preprocessing for phishing URL model development."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS, extract_feature_frame, normalize_url


LABEL_MAP = {
    "1": 1,
    "0": 0,
    "phishing": 1,
    "malicious": 1,
    "bad": 1,
    "unsafe": 1,
    "credential_harvesting": 1,
    "legitimate": 0,
    "benign": 0,
    "good": 0,
    "safe": 0,
}


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for transforming raw URL records into model features."""

    url_column: str = "url"
    label_column: str = "label"
    drop_duplicate_urls: bool = True
    minimum_url_length: int = 4
    maximum_url_length: int = 4096
    outlier_iqr_multiplier: float = 1.5


@dataclass
class PreprocessedDataset:
    """Container returned by the preprocessing pipeline."""

    features: pd.DataFrame
    labels: pd.Series
    cleaned_records: pd.DataFrame
    quality_report: Dict[str, Any]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset with URL and label columns."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def encode_label(value: Any) -> int:
    """Convert provider labels into the project binary convention."""

    if pd.isna(value):
        raise ValueError("Missing label value")

    if isinstance(value, (int, np.integer)):
        if int(value) in {0, 1}:
            return int(value)
    if isinstance(value, (float, np.floating)) and not np.isnan(value):
        if int(value) in {0, 1} and float(value) == float(int(value)):
            return int(value)

    key = str(value).strip().lower()
    if key in LABEL_MAP:
        return LABEL_MAP[key]
    raise ValueError(f"Unsupported label value: {value!r}")


def validate_required_columns(frame: pd.DataFrame, config: PreprocessingConfig) -> None:
    """Ensure the raw dataset contains the columns needed for training."""

    missing = [column for column in [config.url_column, config.label_column] if column not in frame.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Dataset is missing required column(s): {joined}")


def normalize_records(frame: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
    """Normalize URL text and labels while keeping useful metadata columns."""

    validate_required_columns(frame, config)
    normalized = frame.copy()
    normalized = normalized[normalized[config.url_column].notna()]
    normalized[config.url_column] = normalized[config.url_column].astype(str).str.strip()
    normalized = normalized[normalized[config.url_column].str.len() >= config.minimum_url_length]
    normalized = normalized[normalized[config.url_column].str.len() <= config.maximum_url_length]
    normalized["normalized_url"] = normalized[config.url_column].map(normalize_url)
    normalized = normalized[normalized["normalized_url"].str.len() > 0]
    normalized[config.label_column] = normalized[config.label_column].map(encode_label)
    return normalized.reset_index(drop=True)


def remove_duplicate_urls(frame: pd.DataFrame, config: PreprocessingConfig) -> tuple[pd.DataFrame, int]:
    """Remove duplicate normalized URLs."""

    if not config.drop_duplicate_urls:
        return frame, 0
    before = len(frame)
    deduplicated = frame.drop_duplicates(subset=["normalized_url"], keep="first").reset_index(drop=True)
    return deduplicated, before - len(deduplicated)


def analyze_outliers(feature_frame: pd.DataFrame, multiplier: float = 1.5) -> Dict[str, int]:
    """Count IQR outliers for each numeric feature."""

    outliers: Dict[str, int] = {}
    for column in feature_frame.columns:
        values = pd.to_numeric(feature_frame[column], errors="coerce").dropna()
        if values.empty:
            outliers[column] = 0
            continue
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            outliers[column] = 0
            continue
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        outliers[column] = int(((values < lower) | (values > upper)).sum())
    return outliers


def class_distribution(labels: Iterable[int]) -> Dict[str, int]:
    """Return label counts using readable keys."""

    series = pd.Series(labels)
    return {
        "legitimate": int((series == 0).sum()),
        "phishing": int((series == 1).sum()),
    }


def build_numeric_preprocessor(with_scaling: bool = True) -> Pipeline:
    """Create a reusable numeric preprocessing pipeline for model training."""

    steps = [("imputer", SimpleImputer(strategy="median"))]
    if with_scaling:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def preprocess_dataset(
    frame: pd.DataFrame,
    config: Optional[PreprocessingConfig] = None,
) -> PreprocessedDataset:
    """Validate raw records and return feature matrix, labels, and quality data."""

    config = config or PreprocessingConfig()
    raw_count = len(frame)
    cleaned = normalize_records(frame, config)
    cleaned, duplicate_count = remove_duplicate_urls(cleaned, config)

    metadata = cleaned.drop(columns=[config.url_column, config.label_column], errors="ignore")
    feature_frame = extract_feature_frame(cleaned[config.url_column].tolist(), metadata)
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
    feature_frame = feature_frame.reindex(columns=FEATURE_COLUMNS)
    labels = cleaned[config.label_column].astype(int)

    missing_by_feature = feature_frame.isna().sum().astype(int).to_dict()
    outliers = analyze_outliers(feature_frame, config.outlier_iqr_multiplier)
    distribution = class_distribution(labels)

    if distribution["legitimate"] == 0 or distribution["phishing"] == 0:
        raise ValueError("Training data must contain both legitimate and phishing records")

    quality_report: Dict[str, Any] = {
        "raw_records": int(raw_count),
        "cleaned_records": int(len(cleaned)),
        "removed_records": int(raw_count - len(cleaned)),
        "duplicate_urls_removed": int(duplicate_count),
        "class_distribution": distribution,
        "missing_values_by_feature": missing_by_feature,
        "outlier_counts_by_feature": outliers,
        "feature_columns": FEATURE_COLUMNS,
    }

    return PreprocessedDataset(
        features=feature_frame,
        labels=labels.reset_index(drop=True),
        cleaned_records=cleaned.reset_index(drop=True),
        quality_report=quality_report,
    )


def preprocess_csv(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    report_path: Optional[str | Path] = None,
    config: Optional[PreprocessingConfig] = None,
) -> PreprocessedDataset:
    """Load, preprocess, and optionally write a processed feature matrix."""

    dataset = preprocess_dataset(load_dataset(input_path), config)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_frame = dataset.cleaned_records.copy()
        for column in dataset.features.columns:
            output_frame[column] = dataset.features[column]
        output_frame.to_csv(output_path, index=False)

    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(dataset.quality_report, indent=2), encoding="utf-8")

    return dataset
