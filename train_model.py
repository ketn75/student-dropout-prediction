"""
Student Dropout and Academic Success Prediction
Downloads UCI dataset 697, trains supervised classifiers, and saves evaluation outputs.

IMPORTANT:
The UCI dataset contains 4,424 records from a higher-education institution.
It is NOT UEAB student data. Do not describe results as UEAB-specific.
"""
from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


def main():
    print("Downloading UCI Student Dropout and Academic Success dataset (ID 697)...")
    dataset = fetch_ucirepo(id=697)
    X = dataset.data.features.copy()
    y_frame = dataset.data.targets.copy()

    if y_frame.shape[1] < 1:
        raise ValueError("The dataset did not return a target column.")
    y = y_frame.iloc[:, 0].astype(str).str.strip()

    # Basic cleaning
    X = X.replace([float("inf"), float("-inf")], pd.NA)
    X = X.apply(pd.to_numeric, errors="coerce")
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]
    X = X.drop_duplicates()
    y = y.loc[X.index]

    print(f"Records after cleaning: {len(X)}")
    print(f"Predictor columns: {X.shape[1]}")
    print("Class distribution:")
    print(y.value_counts().to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    numeric_features = list(X.columns)
    prep_tree = ColumnTransformer(
        [("numeric", SimpleImputer(strategy="median"), numeric_features)],
        remainder="drop"
    )
    prep_linear = ColumnTransformer(
        [("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), numeric_features)],
        remainder="drop"
    )

    models = {
        "Decision Tree": Pipeline([
            ("preprocess", prep_tree),
            ("classifier", DecisionTreeClassifier(
                random_state=RANDOM_STATE, class_weight="balanced"
            ))
        ]),
        "Random Forest": Pipeline([
            ("preprocess", prep_tree),
            ("classifier", RandomForestClassifier(
                n_estimators=300, random_state=RANDOM_STATE,
                class_weight="balanced", n_jobs=-1
            ))
        ]),
        "Logistic Regression": Pipeline([
            ("preprocess", prep_linear),
            ("classifier", LogisticRegression(
                max_iter=3000, class_weight="balanced"
            ))
        ])
    }

    results = []
    fitted = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        scores = {
            "model": name,
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision_macro": float(precision_score(y_test, pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, pred, average="weighted", zero_division=0))
        }
        results.append(scores)
        fitted[name] = (model, pred)
        print(scores)

    # Select based on macro F1 to account for all three outcome classes.
    results_df = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
    results_df.to_csv(OUT / "model_comparison.csv", index=False)
    best_name = results_df.iloc[0]["model"]
    best_model, best_pred = fitted[best_name]

    report = classification_report(y_test, best_pred, zero_division=0)
    (OUT / "classification_report.txt").write_text(
        f"Selected model: {best_name}\n\n{report}", encoding="utf-8"
    )
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, best_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=30)
    ax.set_title(f"Confusion Matrix - {best_name}")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    joblib.dump(best_model, OUT / "best_student_dropout_model.joblib")
    metadata = {
        "dataset": "UCI Predict Students' Dropout and Academic Success (ID 697)",
        "dataset_doi": "10.24432/C5MC89",
        "records_after_cleaning": int(len(X)),
        "predictor_count": int(X.shape[1]),
        "train_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "target_classes": labels,
        "selected_model": best_name,
        "selection_metric": "macro F1-score",
        "test_metrics": results_df.iloc[0].to_dict(),
        "note": "Dataset is from another higher-education institution, not UEAB."
    }
    (OUT / "results_summary.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    print("\n=== MODEL COMPARISON ===")
    print(results_df.to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print("\nClassification report:\n", report)
    print(f"\nFiles saved in: {OUT.resolve()}")
    print("Use the generated test metrics in the Results and Discussion section.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
