"""Training pipeline for phishing URL detection models."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - exercised only when dependency is unavailable
    XGBClassifier = None

from .evaluation import evaluate_classifier, extract_feature_importance
from .features import FEATURE_COLUMNS
from .preprocessing import PreprocessedDataset, build_numeric_preprocessor, load_dataset, preprocess_dataset


@dataclass(frozen=True)
class ModelSpec:
    """Estimator and tuning information for a candidate classifier."""

    name: str
    pipeline: Pipeline
    parameter_grid: Dict[str, List[Any]]


def build_model_specs(random_state: int = 42) -> List[ModelSpec]:
    """Create the requested model candidates."""

    specs = [
        ModelSpec(
            name="Logistic Regression",
            pipeline=Pipeline(
                steps=[
                    ("preprocess", build_numeric_preprocessor(with_scaling=True)),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            parameter_grid={
                "classifier__C": [0.1, 1.0, 5.0],
                "classifier__solver": ["lbfgs"],
            },
        ),
        ModelSpec(
            name="Decision Tree",
            pipeline=Pipeline(
                steps=[
                    ("preprocess", build_numeric_preprocessor(with_scaling=False)),
                    (
                        "classifier",
                        DecisionTreeClassifier(
                            class_weight="balanced",
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            parameter_grid={
                "classifier__max_depth": [4, 8, 12, None],
                "classifier__min_samples_leaf": [1, 3, 5],
            },
        ),
        ModelSpec(
            name="Random Forest",
            pipeline=Pipeline(
                steps=[
                    ("preprocess", build_numeric_preprocessor(with_scaling=False)),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=200,
                            class_weight="balanced",
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            parameter_grid={
                "classifier__max_depth": [8, 14, None],
                "classifier__min_samples_leaf": [1, 3],
                "classifier__max_features": ["sqrt", "log2"],
            },
        ),
    ]

    if XGBClassifier is not None:
        specs.append(
            ModelSpec(
                name="XGBoost",
                pipeline=Pipeline(
                    steps=[
                        ("preprocess", build_numeric_preprocessor(with_scaling=False)),
                        (
                            "classifier",
                            XGBClassifier(
                                n_estimators=180,
                                learning_rate=0.08,
                                max_depth=5,
                                subsample=0.9,
                                colsample_bytree=0.9,
                                objective="binary:logistic",
                                eval_metric="logloss",
                                random_state=random_state,
                                n_jobs=-1,
                            ),
                        ),
                    ]
                ),
                parameter_grid={
                    "classifier__max_depth": [3, 5],
                    "classifier__learning_rate": [0.05, 0.1],
                    "classifier__subsample": [0.8, 1.0],
                },
            )
        )

    return specs


def choose_cv_splits(labels: pd.Series, maximum: int = 5) -> int:
    """Choose a safe number of stratified folds for the available labels."""

    counts = labels.value_counts()
    if counts.empty:
        return 0
    smallest_class = int(counts.min())
    if smallest_class < 2:
        return 0
    return max(2, min(maximum, smallest_class))


def tune_model(
    spec: ModelSpec,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int,
) -> tuple[Pipeline, Dict[str, Any], Optional[float]]:
    """Tune a model when enough data exists, otherwise fit its defaults."""

    if cv_splits < 2:
        fitted = spec.pipeline.fit(x_train, y_train)
        return fitted, {}, None

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=spec.pipeline,
        param_grid=spec.parameter_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_, dict(search.best_params_), float(search.best_score_)


def cross_validation_summary(
    model: Pipeline,
    features: pd.DataFrame,
    labels: pd.Series,
    cv_splits: int,
) -> Dict[str, Optional[float]]:
    """Run cross validation for the fitted model configuration."""

    if cv_splits < 2:
        return {"mean_f1": None, "std_f1": None}

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, features, labels, scoring="f1", cv=cv, n_jobs=-1)
    return {
        "mean_f1": round(float(np.mean(scores)), 6),
        "std_f1": round(float(np.std(scores)), 6),
    }


def select_best_model(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Select the best candidate using F1-score and phishing recall."""

    return sorted(
        results,
        key=lambda item: (
            item["test_metrics"].get("f1", 0.0),
            item["test_metrics"].get("recall", 0.0),
            -item["test_metrics"].get("confusion_matrix", {}).get("false_positive", 0),
        ),
        reverse=True,
    )[0]


def resolve_test_size(labels: pd.Series, requested_test_size: float) -> float | int:
    """Choose a holdout size that keeps both classes represented when possible."""

    total_records = len(labels)
    class_count = int(labels.nunique())
    class_counts = labels.value_counts()
    if total_records < 4 or class_count < 2 or int(class_counts.min()) < 2:
        raise ValueError("Training data must contain at least two phishing and two legitimate records")

    requested = max(0.1, min(0.5, requested_test_size))
    requested_records = int(np.ceil(total_records * requested))
    minimum_test_records = class_count
    minimum_train_records = class_count
    test_records = max(requested_records, minimum_test_records)

    if total_records - test_records < minimum_train_records:
        test_records = total_records - minimum_train_records
    if test_records < minimum_test_records:
        raise ValueError("Training data is too small for a reliable train/test split")

    return test_records


def train_and_evaluate(
    dataset: PreprocessedDataset,
    output_dir: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train all candidate models and persist the best model."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = dataset.features
    labels = dataset.labels
    split_cv = choose_cv_splits(labels)
    stratify = labels if split_cv >= 2 else None
    effective_test_size = resolve_test_size(labels, test_size)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=effective_test_size,
        random_state=random_state,
        stratify=stratify,
    )

    train_cv = choose_cv_splits(y_train)
    full_cv = choose_cv_splits(labels)
    results: List[Dict[str, Any]] = []

    for spec in build_model_specs(random_state):
        fitted_model, best_params, best_cv_score = tune_model(spec, x_train, y_train, train_cv)
        test_metrics = evaluate_classifier(fitted_model, x_test, y_test)
        cv_summary = cross_validation_summary(fitted_model, features, labels, full_cv)
        feature_importance = extract_feature_importance(fitted_model, FEATURE_COLUMNS)

        results.append(
            {
                "model_name": spec.name,
                "best_params": best_params,
                "tuning_best_f1": round(best_cv_score, 6) if best_cv_score is not None else None,
                "cross_validation": cv_summary,
                "test_metrics": test_metrics,
                "feature_importance": feature_importance,
                "fitted_model": fitted_model,
            }
        )

    best_result = select_best_model(results)
    best_model = best_result["fitted_model"]
    model_path = output_dir / "best_model.joblib"
    metadata_path = output_dir / "training_metadata.json"

    joblib.dump(best_model, model_path)

    serializable_results = []
    for result in results:
        result_copy = {key: value for key, value in result.items() if key != "fitted_model"}
        serializable_results.append(result_copy)

    metadata: Dict[str, Any] = {
        "best_model_name": best_result["model_name"],
        "model_path": str(model_path),
        "feature_columns": FEATURE_COLUMNS,
        "quality_report": dataset.quality_report,
        "test_size": effective_test_size,
        "model_results": serializable_results,
        "xgboost_available": XGBClassifier is not None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return metadata


def train_from_csv(
    input_path: str | Path,
    output_dir: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Load a CSV file, preprocess it, train models, and save artifacts."""

    frame = load_dataset(input_path)
    dataset = preprocess_dataset(frame)
    return train_and_evaluate(
        dataset=dataset,
        output_dir=output_dir,
        test_size=test_size,
        random_state=random_state,
    )


def main() -> None:
    """Command-line entry point for model training."""

    parser = argparse.ArgumentParser(description="Train phishing URL detection models")
    parser.add_argument("--input", required=True, help="Path to unified URL dataset CSV")
    parser.add_argument("--output-dir", default="models", help="Directory for model artifacts")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout test split size")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    metadata = train_from_csv(
        input_path=args.input,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
