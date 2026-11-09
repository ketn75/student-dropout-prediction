# Part Two: Results and Discussion

The UCI dataset contained **4,424 usable student records** and **36 predictor variables** after the reported cleaning process. The data was divided using a stratified 80:20 train-test split: 3,539 records for training and 885 for testing. Decision Tree, Random Forest, and Logistic Regression were trained and evaluated.

Random Forest achieved the strongest macro F1-score and was selected as the preferred model. On the held-out test set, it achieved **75.59% accuracy**, **71.28% macro precision**, **70.83% macro recall**, **70.78% macro F1-score**, and **75.91% weighted F1-score**.

| Classifier | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| Decision Tree | 65.99% | 60.70% | 60.90% | 60.69% |
| Random Forest | 75.59% | 71.28% | 70.83% | 70.78% |
| Logistic Regression | 72.88% | 70.49% | 70.47% | 69.27% |

For Random Forest, the class-wise results were:

| Outcome | Precision | Recall | F1-score | Test support |
|---|---:|---:|---:|---:|
| Dropout | 0.83 | 0.71 | 0.76 | 284 |
| Enrolled | 0.48 | 0.56 | 0.52 | 159 |
| Graduate | 0.83 | 0.86 | 0.84 | 442 |

The results indicate that the objective was **partially achieved**. The model performed better at identifying graduates and dropout cases than students who remained enrolled. The Enrolled class F1-score of 0.52 shows that this category was more difficult to predict and that the model requires further improvement. Dropout recall of 0.71 also means that some students who actually dropped out were not identified by the model.

**Dataset limitation:** The UCI dataset was collected from a higher-education institution outside Kenya. It is not UEAB data. The results therefore demonstrate a proof of concept and must not be interpreted as performance validated on UEAB students. A real UEAB application would require institutional approval, an appropriately anonymised local dataset, further validation, and privacy and fairness safeguards.

# Summary and Conclusion

This project developed a supervised machine learning approach for classifying student academic outcomes into Dropout, Enrolled, and Graduate. Decision Tree, Random Forest, and Logistic Regression were compared using a stratified hold-out test set and standard classification metrics. Random Forest achieved the highest macro F1-score of 70.78% and was selected as the preferred model. The project demonstrates how data mining could support early identification of students who may need academic or other assistance. However, the lower performance for the Enrolled class and the use of non-UEAB data limit immediate practical application.

Future work should include collecting authorised and anonymised UEAB data, validating the model locally, using cross-validation and hyperparameter tuning, testing class-imbalance methods, assessing fairness across student groups, and monitoring performance over time. Predictions should support—not replace—professional judgement and student support services.

**GitHub repository:** https://github.com/ketn75/student-dropout-prediction
