# Student Academic Outcome Prediction

## Project overview
This project applies supervised machine learning to classify higher-education student outcomes into three classes:
- Dropout
- Enrolled
- Graduate

It compares Decision Tree, Random Forest, and Logistic Regression classifiers. The preferred model is selected using macro F1-score so that performance across all outcome classes is considered.

## Dataset
The experiment uses the **UCI Predict Students' Dropout and Academic Success** dataset:
- 4,424 student records
- 36 predictor features
- Three outcome classes
- UCI dataset ID: 697
- DOI: https://doi.org/10.24432/C5MC89
- Dataset page: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

**Important limitation:** This dataset comes from a higher-education institution outside Kenya. It is not UEAB data. The project is a proposed early-warning application for the University of Eastern Africa, Baraton; its measured performance must not be represented as validated UEAB performance. A real UEAB deployment would require institutional approval and an appropriately anonymised local dataset.

## Requirements
Python 3.10 or later is recommended.

### Windows PowerShell setup
```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## Run the experiment
```powershell
python train_model.py
```

The script retrieves dataset 697 using the `ucimlrepo` package, prepares the data, performs a stratified 80:20 train-test split, trains the three classifiers, evaluates them, and selects the best model by macro F1-score.

## Outputs
After running the script, the `outputs/` folder is populated with:
- `model_comparison.csv` — metrics for the compared classifiers
- `classification_report.txt` — per-class precision, recall, F1-score and support
- `confusion_matrix.png` — confusion-matrix visualisation
- `results_summary.json` — dataset and evaluation summary
- `best_student_dropout_model.joblib` — saved best model

The generated model and result files are not committed here; run the script to recreate them. Do not upload private student records or confidential institutional information.

## Reported experiment results
The run used for the accompanying coursework report produced the following test metrics:

| Classifier | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| Decision Tree | 65.99% | 60.70% | 60.90% | 60.69% |
| Random Forest | 75.59% | 71.28% | 70.83% | 70.78% |
| Logistic Regression | 72.88% | 70.49% | 70.47% | 69.27% |

Random Forest was selected. Its per-class F1-scores were 76% for Dropout, 52% for Enrolled, and 84% for Graduate. The weaker Enrolled result is an important limitation and should be considered before any practical use.

## Project files
- `train_model.py` — model training and evaluation
- `requirements.txt` — Python dependencies
- `REPORT_RESULTS_TEMPLATE.md` — report-writing guide
- `README.md` — setup, dataset, methods and limitations

## Academic and ethical note
This is a predictive proof of concept, not a deployed student monitoring system. Predictions should support—not replace—professional academic advising. Local validation, fairness assessment, privacy safeguards, and institutional approval are necessary before operational use.
