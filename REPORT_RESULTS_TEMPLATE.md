# Part Two: Results and Discussion

## 1. Exploratory data analysis

The enhanced project includes an exploratory data analysis stage before model fitting. The training script reports dataset dimensions, target-class distribution, feature data types, unique-value counts and missing-value totals. It generates descriptive statistics and a class distribution chart for Dropout, Enrolled and Graduate.

Pearson correlation analysis is restricted to quantitative fields such as grades, age, curricular-unit counts and macroeconomic indicators. Integer-coded nominal fields, including course, nationality and occupation categories, are excluded because their numeric codes do not imply meaningful order or distance. Correlations describe association and do not establish causation.

After running `python train_model.py`, use the following outputs to write the actual observations:
- `outputs/dataset_exploration.json`
- `outputs/class_distribution.csv` and `outputs/class_distribution.png`
- `outputs/missing_values.csv`
- `outputs/descriptive_statistics.csv`
- `outputs/quantitative_feature_correlations.csv` and `outputs/correlation_heatmap.png`

Do not invent trends or claims that are not visible in the generated data.

## 2. Model development and validation

The target contains three known student outcomes: Dropout, Enrolled and Graduate. The enhanced pipeline uses Decision Tree, Random Forest and Logistic Regression.

The data is split using a stratified 80:20 training-test split. The training split is evaluated with five-fold stratified cross-validation. Preprocessing, including median imputation and scaling for Logistic Regression, is contained inside each scikit-learn pipeline so the transformations are fitted within each training fold. The algorithm is selected using mean cross-validation macro F1 on the training records. The held-out test split is then used for final evaluation.

After rerunning the enhanced training script, report the measured values from:
- `outputs/model_comparison.csv` — hold-out metrics and cross-validation metrics.
- `outputs/cross_validation_summary.csv` — mean and standard deviation of cross-validation accuracy and macro F1.
- `outputs/classification_report.txt` — per-class precision, recall and F1.
- `outputs/confusion_matrix.png` — correct and incorrect predictions by class.

### Earlier experiment (historical baseline only)

The earlier coursework experiment, before the addition of cross-validation and the expanded exploratory analysis, reported the following hold-out results:

| Classifier | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| Decision Tree | 65.99% | 60.70% | 60.90% | 60.69% |
| Random Forest | 75.59% | 71.28% | 70.83% | 70.78% |
| Logistic Regression | 72.88% | 70.49% | 70.47% | 69.27% |

In that earlier experiment, Random Forest achieved the strongest macro F1. Its earlier class-wise F1-scores were 0.76 for Dropout, 0.52 for Enrolled and 0.84 for Graduate. These figures are retained as historical baseline results; **they are not the results of the enhanced five-fold cross-validation workflow**. Replace or supplement them with the new measured values after rerunning the enhanced script.

## 3. Feature importance

The enhanced script calculates model-agnostic permutation feature importance for the selected model on the held-out test records. A feature is more important in this analysis when shuffling its values reduces test macro F1 more substantially. The output is saved to:
- `outputs/feature_importance.csv`
- `outputs/feature_importance.png`

Discuss the leading features only after inspecting the generated values. Permutation importance describes the model's predictive reliance on a variable; it does not prove that the variable causes dropout or graduation.

## 4. Discussion and limitations

The earlier baseline showed stronger performance for Graduate and Dropout than for Enrolled. In particular, the Enrolled class F1-score was 0.52 in the earlier run, showing that this category was more difficult to identify. The enhanced experiment should be discussed using its newly generated test metrics and class-specific report.

**Dataset limitation:** The UCI dataset was collected from a higher-education institution outside Kenya. It is not UEAB data. The results demonstrate a proof of concept and must not be interpreted as performance validated on UEAB students. A real UEAB application would require institutional approval, an appropriately anonymised local dataset, local validation, privacy safeguards, fairness assessment and ongoing monitoring.

## 5. Summary and conclusion

This project develops a supervised machine learning approach for classifying student academic outcomes into Dropout, Enrolled and Graduate. The enhanced workflow strengthens the original project through exploratory data analysis, quantitative-feature correlation analysis, five-fold stratified cross-validation, hold-out test evaluation, confusion-matrix review and permutation feature importance. State the final selected algorithm and numerical performance only after running the enhanced script and checking the generated output files.

Future work should include collecting authorised and anonymised UEAB data, validating the model locally, tuning model parameters, testing class-imbalance methods, assessing fairness across student groups and monitoring performance over time. Predictions should support—not replace—professional judgement and student support services.

**GitHub repository:** https://github.com/ketn75/student-dropout-prediction
