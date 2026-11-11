"""Local Flask dashboard for the trained student-outcome model."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for

BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "outputs"
MODEL_PATH = OUT / "best_student_dropout_model.joblib"
app = Flask(__name__)
app.secret_key = "change-this-for-local-development"


def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def dashboard_data():
    summary = {}
    comparison = []
    classes = []
    example = {}
    try:
        summary = json.loads((OUT / "results_summary.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    try:
        comparison = pd.read_csv(OUT / "model_comparison.csv").to_dict(orient="records")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        pass
    try:
        example = json.loads((OUT / "example_student_input.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    if summary:
        classes = summary.get("target_classes", [])
    return summary, comparison, classes, example


@app.route("/", methods=["GET", "POST"])
def index():
    model = load_model()
    summary, comparison, classes, example = dashboard_data()
    feature_names = list(getattr(model, "feature_names_in_", [])) if model else []
    form_values = example.copy()

    if request.method == "POST":
        if model is None:
            flash("Train the model first by running python train_model.py.", "error")
            return redirect(url_for("index"))

        values = {}
        errors = []
        for feature in feature_names:
            raw = request.form.get(feature, "").strip()
            if raw == "":
                errors.append(f"Enter a value for {feature}.")
                continue
            try:
                number = float(raw)
                if not pd.notna(number):
                    raise ValueError
                values[feature] = number
            except ValueError:
                errors.append(f"{feature} must be a valid number.")

        if errors:
            for error in errors[:5]:
                flash(error, "error")
            form_values = request.form.to_dict()
        else:
            row = pd.DataFrame([values], columns=feature_names)
            prediction = str(model.predict(row)[0])
            probabilities = {}
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(row)[0]
                probabilities = {
                    str(label): round(float(prob) * 100, 2)
                    for label, prob in zip(model.classes_, proba)
                }
            return render_template(
                "index.html", model_ready=True, feature_names=feature_names,
                form_values=values, summary=summary, comparison=comparison,
                classes=classes, prediction=prediction, probabilities=probabilities,
                example=example
            )

    return render_template(
        "index.html", model_ready=model is not None, feature_names=feature_names,
        form_values=form_values, summary=summary, comparison=comparison,
        classes=classes, prediction=None, probabilities={}, example=example
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
