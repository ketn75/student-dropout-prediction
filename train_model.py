"""Student Academic Outcome Prediction with exploratory analysis and evaluation.

Dataset: UCI Predict Students' Dropout and Academic Success (ID 697).
It is not UEAB data; this project is a proof of concept.
"""
from __future__ import annotations

import json
from pathlib import Path
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, ConfusionMatrixDisplay,
    confusion_matrix, f1_score, precision_score, recall_score
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
CV_FOLDS = 5
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# Pearson correlations are limited to quantitative/count/grade measures.
# Nominal category codes (e.g. course, nationality, occupation) are excluded.
QUANTITATIVE_FEATURES = [
    "Previous qualification (grade)", "Admission grade", "Age at enrollment",
    "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate", "Inflation rate", "GDP",
]


def create_eda_outputs(X: pd.DataFrame, y: pd.Series) -> dict:
    """Save class balance, data quality, descriptive statistics and correlations."""
    counts = y.value_counts().rename_axis("outcome").reset_index(name="records")
    counts["percentage"] = counts["records"] / len(y) * 100
    counts.to_csv(OUT / "class_distribution.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ordered = counts.sort_values("records")
    bars = ax.barh(ordered["outcome"], ordered["records"])
    ax.set(title="Student outcome class distribution", xlabel="Number of records", ylabel="Outcome")
    ax.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, ordered["records"]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {int(value):,}", va="center")
    fig.tight_layout()
    fig.savefig(OUT / "class_distribution.png", dpi=180)
    plt.close(fig)

    missing = X.isna().sum().rename("missing_values").to_frame()
    missing["missing_percentage"] = missing["missing_values"] / len(X) * 100
    missing.to_csv(OUT / "missing_values.csv")

    profile = pd.DataFrame({
        "feature": X.columns,
        "data_type": [str(X[c].dtype) for c in X.columns],
        "unique_values": [int(X[c].nunique(dropna=True)) for c in X.columns],
        "missing_values": [int(X[c].isna().sum()) for c in X.columns],
    })
    profile.to_csv(OUT / "feature_profile.csv", index=False)
    X.describe().T.to_csv(OUT / "descriptive_statistics.csv")

    corr_features = [c for c in QUANTITATIVE_FEATURES if c in X.columns]
    if len(corr_features) >= 2:
        corr = X[corr_features].corr(method="pearson")
        corr.to_csv(OUT / "quantitative_feature_correlations.csv")
        size = max(10, min(17, 0.72 * len(corr_features)))
        fig, ax = plt.subplots(figsize=(size, size * 0.78))
        im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr_features)))
        ax.set_yticks(range(len(corr_features)))
        ax.set_xticklabels(corr_features, rotation=75, ha="right", fontsize=7)
        ax.set_yticklabels(corr_features, fontsize=7)
        ax.set_title("Pearson correlation: quantitative student features")
        fig.colorbar(im, ax=ax, label="Correlation")
        fig.tight_layout()
        fig.savefig(OUT / "correlation_heatmap.png", dpi=180)
        plt.close(fig)

    details = {
        "records": int(len(X)),
        "predictor_features": int(X.shape[1]),
        "target_classes": [str(v) for v in sorted(y.unique())],
        "features_with_missing_values": int((X.isna().sum() > 0).sum()),
        "total_missing_predictor_values": int(X.isna().sum().sum()),
        "class_distribution": counts.to_dict(orient="records"),
        "correlation_features": corr_features,
        "correlation_note": "Pearson correlation only describes association; nominal integer category codes were excluded."
    }
    (OUT / "dataset_exploration.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    return details


def build_models(columns: list[str]) -> dict[str, Pipeline]:
    tree_preprocess = ColumnTransformer(
        [("numeric", SimpleImputer(strategy="median"), columns)],
        remainder="drop"
    )
    linear_preprocess = ColumnTransformer(
        [("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), columns)],
        remainder="drop"
    )
    return {
        "Decision Tree": Pipeline([
            ("preprocess", tree_preprocess),
            ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"))
        ]),
        "Random Forest": Pipeline([
            ("preprocess", tree_preprocess),
            ("classifier", RandomForestClassifier(
                n_estimators=300, random_state=RANDOM_STATE,
                class_weight="balanced", n_jobs=-1
            ))
        ]),
        "Logistic Regression": Pipeline([
            ("preprocess", linear_preprocess),
            ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced"))
        ]),
    }


def main() -> None:
    print("Downloading UCI Student Dropout and Academic Success dataset (ID 697)...")
    try:
        dataset = fetch_ucirepo(id=697)
    except Exception as exc:
        raise RuntimeError(
            "Unable to download UCI dataset 697. Check the internet connection and retry. "
            f"Original error: {exc}"
        ) from exc

    X = dataset.data.features.copy()
    target = dataset.data.targets
    if target is None or target.shape[1] == 0:
        raise ValueError("Dataset target column was not returned.")
    y = target.iloc[:, 0].astype("string").str.strip()

    X = X.replace([np.inf, -np.inf], np.nan).apply(pd.to_numeric, errors="coerce")
    valid = y.notna()
    X, y = X.loc[valid].copy(), y.loc[valid].astype(str).copy()

    # Remove only exact repeated feature+target rows; keep repeated profiles with
    # different outcomes so that potentially conflicting records are not hidden.
    check = X.copy()
    check["__target__"] = y
    duplicate_mask = check.duplicated(keep="first")
    duplicates_removed = int(duplicate_mask.sum())
    X = X.loc[~duplicate_mask].copy()
    y = y.loc[X.index].copy()

    print(f"Records after cleaning: {len(X):,}")
    print(f"Predictor columns: {X.shape[1]}")
    print(f"Exact duplicates removed: {duplicates_removed}")
    print("Class distribution:\n", y.value_counts().to_string())

    print("\nCreating exploratory data analysis outputs...")
    eda = create_eda_outputs(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training records: {len(X_train):,}; test records: {len(X_test):,}")

    models = build_models(list(X.columns))
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }

    rows = []
    fitted = {}
    for name, model in models.items():
        print(f"\nRunning {CV_FOLDS}-fold cross-validation: {name}")
        cv_result = cross_validate(
            model, X_train, y_train, cv=cv, scoring=scoring,
            n_jobs=1, return_train_score=False
        )
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        row = {
            "model": name,
            "accuracy": float(accuracy_score(y_test, prediction)),
            "precision_macro": float(precision_score(y_test, prediction, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, prediction, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, prediction, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, prediction, average="weighted", zero_division=0)),
            "cv_accuracy_mean": float(cv_result["test_accuracy"].mean()),
            "cv_accuracy_std": float(cv_result["test_accuracy"].std(ddof=1)),
            "cv_f1_macro_mean": float(cv_result["test_f1_macro"].mean()),
            "cv_f1_macro_std": float(cv_result["test_f1_macro"].std(ddof=1)),
        }
        rows.append(row)
        fitted[name] = (model, prediction)
        print(f"CV macro F1: {row['cv_f1_macro_mean']:.4f} +/- {row['cv_f1_macro_std']:.4f}")

    results = pd.DataFrame(rows).sort_values("cv_f1_macro_mean", ascending=False)
    results.to_csv(OUT / "model_comparison.csv", index=False)
    results[["model", "cv_accuracy_mean", "cv_accuracy_std", "cv_f1_macro_mean", "cv_f1_macro_std"]].to_csv(
        OUT / "cross_validation_summary.csv", index=False
    )

    best_name = str(results.iloc[0]["model"])
    best_model, best_prediction = fitted[best_name]
    report = classification_report(y_test, best_prediction, zero_division=0)
    (OUT / "classification_report.txt").write_text(
        f"Selected by five-fold training CV macro F1: {best_name}\n\n{report}",
        encoding="utf-8"
    )

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, best_prediction, labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    display.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=25, colorbar=False)
    ax.set_title(f"Confusion Matrix - {best_name}")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    print("\nCalculating permutation feature importance...")
    importance = permutation_importance(
        best_model, X_test, y_test, scoring="f1_macro",
        n_repeats=5, random_state=RANDOM_STATE, n_jobs=1
    )
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(OUT / "feature_importance.csv", index=False)

    top = importance_df.head(15).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    ax.axvline(0, linewidth=0.8, color="black")
    ax.set(title=f"Permutation Feature Importance - {best_name}",
           xlabel="Decrease in test macro F1 when feature is shuffled",
           ylabel="Feature")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=180)
    plt.close(fig)

    pred_frame = pd.DataFrame({
        "actual_outcome": y_test,
        "predicted_outcome": pd.Series(best_prediction, index=y_test.index),
    }).sort_index()
    pred_frame.to_csv(OUT / "test_predictions.csv", index_label="source_row")

    joblib.dump(best_model, OUT / "best_student_dropout_model.joblib")
    # One real row from the public test split is stored only as an illustrative
    # dashboard input. It is not an identified student record.
    example = X_test.iloc[0].to_dict()
    (OUT / "example_student_input.json").write_text(
        json.dumps({str(k): float(v) for k, v in example.items()}, indent=2),
        encoding="utf-8"
    )

    selected = results.iloc[0].to_dict()
    summary = {
        "dataset": "UCI Predict Students' Dropout and Academic Success (ID 697)",
        "dataset_doi": "10.24432/C5MC89",
        "records_after_cleaning": int(len(X)),
        "exact_duplicate_records_removed": duplicates_removed,
        "predictor_count": int(X.shape[1]),
        "train_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "target_classes": labels,
        "selected_model": best_name,
        "selection_metric": "mean five-fold cross-validation macro F1 on training data",
        "test_metrics": selected,
        "cross_validation_folds": CV_FOLDS,
        "eda": eda,
        "top_features_by_permutation_importance": importance_df.head(10).to_dict(orient="records"),
        "interpretation_note": "Permutation importance and correlation are not evidence of causation.",
        "dataset_limitation": "Dataset is from another higher-education institution, not UEAB."
    }
    (OUT / "results_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== MODEL COMPARISON (ranked by training cross-validation macro F1) ===")
    print(results.to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print("\nClassification report:\n", report)
    print("\nTop permutation importance features:\n", importance_df.head(10).to_string(index=False))
    print(f"\nAnalysis files saved in: {OUT}")
    print("The dataset is public UCI data, not UEAB student data.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
