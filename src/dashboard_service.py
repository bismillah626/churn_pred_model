"""Analytics data preparation for the Flask dashboard.

The dashboard intentionally receives plain JSON rather than Plotly figures. This
keeps the browser interactive (filters can redraw every chart instantly) while
keeping the data science work in a testable Python module.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler
from sklearn.tree import DecisionTreeClassifier


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "credit-card-customer-churn.csv"

MODEL_FEATURES = [
    "CreditScore",
    "Age",
    "IsActiveMember",
    "Tenure",
    "Gender",
    "Geography",
    "NumOfProducts",
]
NUMERIC_FEATURES = ["CreditScore", "Age", "IsActiveMember", "Tenure"]
CATEGORICAL_FEATURES = ["Gender", "Geography", "NumOfProducts"]
DISPLAY_COLUMNS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Exited",
]


def _json_number(value: Any) -> Any:
    """Convert numpy scalars and non-finite values into JSON-safe values."""

    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 6)
    return value


def _safe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return a compact JSON-safe representation of a dataframe."""

    return [
        {column: _json_number(value) for column, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]


def _product_label(value: Any) -> str:
    labels = {1: "One", 2: "Two", 3: "Three", 4: "Four"}
    try:
        return labels.get(int(value), str(value))
    except (TypeError, ValueError):
        return str(value)


def _make_encoder() -> OneHotEncoder:
    """Support both the current and older scikit-learn encoder APIs."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _make_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", RobustScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _make_encoder()),
            ("scaler", MinMaxScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def _make_models() -> dict[str, Any]:
    """Use the same family of models as the existing training pipeline."""

    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=1),
        "Decision Tree": DecisionTreeClassifier(random_state=1),
        "Random Forest": RandomForestClassifier(n_estimators=120, random_state=1, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=120, random_state=1, n_jobs=-1),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=1),
        "K-nearest Neighbors": KNeighborsClassifier(),
    }


def _model_scores(model: Any, features: np.ndarray) -> np.ndarray:
    """Return a positive-class score for ROC, PR, and probability charts."""

    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return model.decision_function(features)


def _importance_rows(
    model: Any,
    feature_names: list[str],
    test_features: np.ndarray,
    y_test: pd.Series,
) -> list[dict[str, Any]]:
    """Extract model importance, with permutation importance as a fallback."""

    values: np.ndarray | None = None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_)[0])
    else:
        try:
            from sklearn.inspection import permutation_importance

            result = permutation_importance(
                model,
                test_features,
                y_test,
                n_repeats=3,
                random_state=1,
                scoring="f1",
            )
            values = np.maximum(result.importances_mean, 0)
        except Exception:
            values = np.zeros(len(feature_names))

    rows = []
    for name, importance in zip(feature_names, values):
        clean_name = name.replace("numeric__", "").replace("categorical__", "")
        rows.append({"feature": clean_name, "importance": round(float(importance), 6)})
    return sorted(rows, key=lambda row: row["importance"], reverse=True)[:15]


def _shap_rows(
    model: Any,
    test_features: np.ndarray,
    feature_names: list[str],
    fallback: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    """Generate a small SHAP summary when the installed SHAP version supports it."""

    try:
        import shap

        sample = test_features[: min(250, len(test_features))]
        explainer = shap.TreeExplainer(model)
        raw_values = explainer.shap_values(sample)
        if isinstance(raw_values, list):
            values = np.asarray(raw_values[min(1, len(raw_values) - 1)])
        else:
            values = np.asarray(raw_values)
            if values.ndim == 3:
                values = values[:, :, min(1, values.shape[2] - 1)]
        if values.ndim != 2:
            raise ValueError("Unsupported SHAP output shape")
        mean_abs = np.abs(values).mean(axis=0)
        rows = [
            {
                "feature": name.replace("numeric__", "").replace("categorical__", ""),
                "importance": round(float(value), 6),
            }
            for name, value in zip(feature_names, mean_abs)
        ]
        return sorted(rows, key=lambda row: row["importance"], reverse=True)[:15], True
    except Exception:
        # The feature-importance chart is still useful when SHAP is unavailable
        # or incompatible with a serialized estimator.
        return fallback, False


def _build_model_analytics(data: pd.DataFrame) -> dict[str, Any]:
    """Train dashboard-only evaluation copies and serialize their diagnostics."""

    model_data = data[MODEL_FEATURES + ["Exited"]].copy()
    model_data["NumOfProducts"] = model_data["NumOfProducts"].map(_product_label)
    model_data["Exited"] = model_data["Exited"].astype(int)

    X = model_data[MODEL_FEATURES]
    y = model_data["Exited"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    preprocessor = _make_preprocessor()
    train_features = preprocessor.fit_transform(X_train)
    test_features = preprocessor.transform(X_test)

    # Match the production pipeline's SMOTE training strategy when available.
    try:
        from imblearn.over_sampling import SMOTE

        train_features, y_train = SMOTE(random_state=1).fit_resample(train_features, y_train)
    except Exception:
        pass

    feature_names = list(preprocessor.get_feature_names_out())
    metrics: list[dict[str, Any]] = []
    roc_curves: list[dict[str, Any]] = []
    pr_curves: list[dict[str, Any]] = []
    confusion_matrices: dict[str, list[list[int]]] = {}
    fitted_models: dict[str, Any] = {}
    probability_series: dict[str, np.ndarray] = {}

    for model_name, model in _make_models().items():
        model.fit(train_features, y_train)
        prediction = model.predict(test_features)
        scores = _model_scores(model, test_features)
        fitted_models[model_name] = model
        probability_series[model_name] = scores
        metrics.append(
            {
                "model": model_name,
                "accuracy": round(float(accuracy_score(y_test, prediction)), 4),
                "precision": round(float(precision_score(y_test, prediction, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, prediction, zero_division=0)), 4),
                "f1": round(float(f1_score(y_test, prediction, zero_division=0)), 4),
                "roc_auc": round(float(roc_auc_score(y_test, scores)), 4),
            }
        )
        fpr, tpr, _ = roc_curve(y_test, scores)
        precision, recall, _ = precision_recall_curve(y_test, scores)
        roc_curves.append({"model": model_name, "fpr": fpr.tolist(), "tpr": tpr.tolist()})
        pr_curves.append(
            {"model": model_name, "precision": precision.tolist(), "recall": recall.tolist()}
        )
        confusion_matrices[model_name] = confusion_matrix(y_test, prediction).tolist()

    best = max(metrics, key=lambda item: item["accuracy"])
    best_name = best["model"]
    best_model = fitted_models[best_name]
    importance = _importance_rows(best_model, feature_names, test_features, y_test)
    shap_summary, shap_available = _shap_rows(best_model, test_features, feature_names, importance)
    probability_scores = probability_series[best_name]

    probabilities = [
        {"probability": round(float(score), 6), "actual": int(actual)}
        for score, actual in zip(probability_scores, y_test)
    ]
    return {
        "model_comparison": metrics,
        "best_model": best_name,
        "roc_curves": roc_curves,
        "pr_curves": pr_curves,
        "confusion_matrices": confusion_matrices,
        "feature_importance": importance,
        "shap_summary": shap_summary,
        "shap_available": shap_available,
        "probabilities": probabilities,
        "test_size": int(len(y_test)),
        "positive_test_count": int(y_test.sum()),
        "average_precision": round(float(average_precision_score(y_test, probability_scores)), 4),
    }


def _build_payload() -> dict[str, Any]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    if "Exited" not in data.columns:
        raise ValueError("The dataset must contain an Exited target column")

    display_data = data[DISPLAY_COLUMNS].copy()
    display_data["NumOfProducts"] = display_data["NumOfProducts"].map(_product_label)
    rows = _safe_records(display_data)
    numeric_summary = data.select_dtypes(include=[np.number])
    model_analytics = _build_model_analytics(data)

    return {
        "rows": rows,
        "columns": DISPLAY_COLUMNS,
        "metadata": {
            "row_count": int(len(data)),
            "churn_rate": round(float(data["Exited"].mean()), 4),
            "avg_credit_score": round(float(data["CreditScore"].mean()), 1),
            "avg_balance": round(float(data["Balance"].mean()), 2),
            "feature_count": len(DISPLAY_COLUMNS) - 1,
            "numeric_features": list(numeric_summary.columns),
            "geographies": sorted(data["Geography"].dropna().unique().tolist()),
            "genders": sorted(data["Gender"].dropna().unique().tolist()),
            "products": sorted({_product_label(value) for value in data["NumOfProducts"]}),
        },
        "model": model_analytics,
    }


@lru_cache(maxsize=1)
def get_dashboard_data() -> dict[str, Any]:
    """Load and cache the analytics payload until the process is restarted."""

    return _build_payload()
