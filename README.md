# Student Academic Outcome Prediction

## Project overview
This project applies supervised machine learning to classify higher-education student outcomes into three classes:
- Dropout
- Enrolled
- Graduate

It compares Decision Tree, Random Forest and Logistic Regression. The enhanced workflow includes exploratory data analysis, data-quality profiling, quantitative-feature correlation analysis, stratified train/test split, five-fold cross-validation on training data, test-set evaluation, confusion matrix and permutation feature importance.

## Enhanced project package
The **complete enhanced VS Code project** is available in the repository as:
[`Student_Dropout_ML_Project_Enhanced.zip`](Student_Dropout_ML_Project_Enhanced.zip)

Download the ZIP, extract it, and open the extracted `Student_Dropout_ML_Project` folder in VS Code. It includes the Flask dashboard, EDA notebook, enhanced training script, styles, templates, requirements and report guide. The enhanced archive is the current complete project package; the root-level starter files are retained for reference.

## Dataset
The experiment uses the **UCI Predict Students' Dropout and Academic Success** dataset:
- 4,424 records in the source dataset
- 36 predictor features
- Three outcome classes
- UCI dataset ID: 697
- DOI: https://doi.org/10.24432/C5MC89
- Dataset page: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

**Important limitation:** This dataset comes from a higher-education institution outside Kenya. It is not UEAB data. The project is a proposed early-warning application for the University of Eastern Africa, Baraton; its measured performance must not be represented as validated UEAB performance. Real UEAB use would require institutional approval, an appropriately anonymised local dataset, local validation, privacy controls and human review.

## Run the enhanced project in VS Code
Python 3.10 or later is recommended. Install the Python and Jupyter extensions in VS Code.

From the extracted project folder, run these commands in PowerShell:

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

Open `student_outcome_eda.ipynb` and run cells from top to bottom for step-by-step exploration.

To start the browser dashboard after model training:

```powershell
python app.py
```

Then open http://127.0.0.1:5000. Do not use VS Code Live Server; Flask runs its own server.

## Enhanced analysis and outputs
The enhanced training script generates:
- Target class distribution table and chart.
- Missing-value report, feature profile and descriptive statistics.
- Pearson correlation table and heatmap for quantitative/count/grade fields only. Integer-coded nominal categories are excluded because their numeric codes do not imply meaningful order.
- Stratified 80:20 train/test split.
- Five-fold stratified cross-validation on training records.
- Decision Tree, Random Forest and Logistic Regression comparisons.
- Hold-out classification report and confusion matrix.
- Permutation feature importance for the selected model.
- Test-set actual-versus-predicted labels.
- A saved model used by the Flask prediction dashboard.

Outputs are saved in `outputs/`. The new pipeline selects a model based on mean cross-validation macro F1 on the training data and uses the test set for final evaluation. Run the enhanced script before quoting new metrics in coursework. Older results in the starter files are historical results from the earlier experiment, not new cross-validation results.

## Interpretation and responsible use
Permutation importance measures how much model performance changes when a feature is shuffled. It does not show that the feature causes a student outcome. Correlations describe association, not causation. This is a predictive proof of concept, not a deployed student monitoring system. Predictions should support—not replace—professional academic advising. Local validation, fairness assessment, privacy safeguards and institutional approval are necessary before operational use.

## GitHub repository
https://github.com/ketn75/student-dropout-prediction
