# Student Academic Outcome Prediction

## Project overview
This project applies supervised machine learning to classify higher-education student outcomes into three classes:
- Dropout
- Enrolled
- Graduate

The enhanced workflow includes exploratory data analysis, data-quality profiling, quantitative-feature correlation analysis, a stratified train/test split, five-fold cross-validation on training data, hold-out test evaluation, a confusion matrix and permutation feature importance. A Flask web dashboard provides an interactive prediction demonstration.

## Dataset
The experiment uses the **UCI Predict Students' Dropout and Academic Success** dataset:
- 4,424 records in the source dataset
- 36 predictor features
- Three outcome classes
- UCI dataset ID: 697
- DOI: https://doi.org/10.24432/C5MC89
- Dataset page: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

**Important limitation:** This dataset comes from a higher-education institution outside Kenya. It is not UEAB data. The project is a proposed early-warning application for the University of Eastern Africa, Baraton; its measured performance must not be represented as validated UEAB performance. Real UEAB use would require institutional approval, an appropriately anonymised local dataset, local validation, privacy controls and human review.

## Run the project in VS Code
Python 3.10 or later is recommended. Install the Python and Jupyter extensions in VS Code.

Clone or download this repository, open the project folder in VS Code, then run these PowerShell commands from the project folder:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

Open `student_outcome_eda.ipynb` and run its cells from top to bottom for step-by-step exploration. The notebook inspects the data and can display output metrics after training.

## Start the browser dashboard
Train the model first using `python train_model.py`. Then start Flask:

```powershell
python app.py
```

Open **http://127.0.0.1:5000** in your browser. Keep the VS Code terminal open while using the dashboard and stop the local server with Ctrl+C. Do not use VS Code Live Server; Flask runs its own local server.

The dashboard's prefilled example comes from a row in the public dataset test split. It is a demonstration input, not a UEAB student record.

## Enhanced analysis and generated outputs
The training script generates:
- `class_distribution.csv` and `class_distribution.png` — counts and percentages for the three outcomes.
- `missing_values.csv`, `feature_profile.csv` and `descriptive_statistics.csv` — data quality and variable descriptions.
- `quantitative_feature_correlations.csv` and `correlation_heatmap.png` — Pearson correlation for quantitative/count/grade fields only. Integer-coded nominal categories are excluded because their numeric codes do not imply meaningful order.
- `model_comparison.csv` and `cross_validation_summary.csv` — three classifier test metrics and five-fold training cross-validation summaries.
- `classification_report.txt` and `confusion_matrix.png` — per-class test performance and confusion matrix.
- `feature_importance.csv` and `feature_importance.png` — permutation feature importance on held-out test data.
- `test_predictions.csv` — actual and predicted labels for test records.
- `dataset_exploration.json` and `results_summary.json` — data exploration and experiment metadata.
- `example_student_input.json` — illustrative form values drawn from a public dataset example row.
- `best_student_dropout_model.joblib` — selected trained model used by Flask.

The model is selected using mean five-fold cross-validation macro F1 on the training split; the hold-out test split is reserved for final evaluation. Rerun the script before quoting new metrics in coursework. Earlier reported accuracy values belong to the previous experiment and are not results from the enhanced workflow.

## Project files
- `train_model.py` — data exploration, cross-validation, training and evaluation
- `student_outcome_eda.ipynb` — step-by-step exploratory data analysis notebook
- `app.py` — Flask prediction dashboard
- `templates/index.html` — dashboard page and prediction form
- `static/style.css` — dashboard styling
- `requirements.txt` — Python dependencies
- `REPORT_RESULTS_TEMPLATE.md` — results, discussion and conclusion guide
- `outputs/` — generated analysis, metrics and saved model

## Interpretation and responsible use
Permutation importance describes how much model performance changes when a feature is shuffled. It does not show that the feature causes a student outcome. Correlations describe association, not causation. This is a predictive proof of concept, not a deployed student monitoring system. Predictions should support—not replace—professional academic advising. Local validation, fairness assessment, privacy safeguards and institutional approval are necessary before operational use.

## GitHub repository
https://github.com/ketn75/student-dropout-prediction
