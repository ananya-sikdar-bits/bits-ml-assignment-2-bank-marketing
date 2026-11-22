# Term Deposit Subscription Classification — UCI Bank Marketing Dataset

## Problem Statement
Direct telemarketing campaigns are widely used by banks to promote term deposit subscriptions, but untargeted calling leads to high operational effort and low conversion rates due to class imbalance. The objective of this project is to build, train, save, and compare five supervised machine learning classification models—**Logistic Regression**, **Decision Tree Classifier**, **K-Nearest Neighbors (kNN)**, **Gaussian Naive Bayes**, and **Random Forest (Ensemble)**—to predict whether a client subscribes to a term deposit (`y`: `1` = `"yes"`, `0` = `"no"`) using demographic, campaign interaction, and macroeconomic features[.].

---

## Dataset Description
- **Dataset:** Bank Marketing Dataset (`bank-additional-full.csv`)
- **Source:** UCI Machine Learning Repository (Dataset ID: 222)
- **Why Selected:** Contains **41,188 rows** and **21 columns** (20 input features + 1 binary target), exceeding the assignment's minimum requirements of 1,000 instances and 12 features while providing real-world `"unknown"` missing markers and class imbalanc.
- **Duplicate Handling:** Removed **12** duplicate rows prior to splitting, leaving **41,176** unique records (`X_train`: 32,940 rows; `X_test`: 8,236 rows exported to `test_data.csv`)
- **Target Variable (`y`):** Mapped `"no"` $\rightarrow$ `0` and `"yes"` $\rightarrow$ `1`
- **Feature Types:**
  - **Numerical features (10):** `age`, `duration`, `campaign`, `pdays`, `previous`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`.
  - **Categorical features (10):** `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome`.
- **Missing Values & Preprocessing:** Standardized `"unknown"` and `"?"` to `NaN` (`default`: 20.87%, `education`: 4.20%, `housing`: 2.40%, `loan`: 2.40%, `job`: 0.80%, `marital`: 0.19%). Applied a leak-free `ColumnTransformer` inside each model `Pipeline` fitted strictly on `X_train` using `SimpleImputer(strategy="median")` + `StandardScaler()` for numerical features and `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")` for categorical features, with a stratified 80:20 train-test split (`random_state=42`).
- **Class Distribution:** Class `0` (`no`) accounts for **36,548** instances (**88.73%**) and Class `1` (`yes`) accounts for **4,640** instances (**11.27%**), preserving the ~88.73% / ~11.27% ratio across train and test splits[.].

---

## GitHub Repository Link
- **Repository URL:** `https://github.com/ananya-sikdar-bits/bits-ml-assignment-2-bank-marketing.git`
- **Expected Structure:**
  ```text
  project-folder/
  ├── ML_Assignment2.ipynb
  ├── requirements.txt
  ├── README.md
  ├── test_data.csv
  └── model/
      ├── logistic_regression.pkl
      ├── decision_tree.pkl
      ├── knn.pkl
      ├── naive_bayes.pkl
      └── random_forest_ensemble.pkl
  ```

---

## Execution Instructions
1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Open and run `ML_Assignment2.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab (`RANDOM_SEED = 42`).
3. The notebook cleans duplicates and missing markers, creates the stratified 80:20 split, saves `test_data.csv`, trains and saves all five model pipelines into `model/`, prints the comparison table, and generates the evaluation charts[., 31].

---

## Model Configurations & Evaluation Metrics Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.8648 | 0.9437 | 0.4504 | 0.9106 | 0.6027 | 0.5801 |
| Decision Tree | 0.8599 | 0.9294 | 0.4410 | 0.9095 | 0.5939 | 0.5713 |
| kNN | 0.9051 | 0.8943 | 0.6148 | 0.4213 | 0.5000 | 0.4594 |
| Naive Bayes | 0.7217 | 0.8470 | 0.2681 | 0.8502 | 0.4076 | 0.3665 |
| Random Forest (Ensemble) | 0.8835 | 0.9503 | 0.4906 | 0.9019 | 0.6355 | 0.6110 |



---

## Graph Representation (Visualizations)
- **Grouped Bar Chart (`metrics_bar_chart.png`):** Compares `Accuracy`, `AUC`, `Precision`, `Recall`, `F1`, and `MCC` across all five classification models[., 12].
- **ROC Curves (`roc_comparison.png`):** Plots True Positive Rate against False Positive Rate with AUC scores for each model relative to a random classifier baseline[., 11].

---

## Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Achieves strong ranking performance (**AUC = 0.9437**) and the highest sensitivity (**Recall = 0.9106**), capturing over 91% of subscribers[., 14]. With `class_weight="balanced"`, it trades Precision (**0.4504**) and Accuracy (**0.8648**) for higher positive recall while keeping **F1 = 0.6027** and **MCC = 0.5801**[., 14]. |
| Decision Tree | Performs closely to Logistic Regression (**AUC = 0.9294**, **Recall = 0.9095**, **F1 = 0.5939**, **MCC = 0.5713**)[., 14]. Constraining `max_depth=10` and `min_samples_leaf=10` with `class_weight="balanced"` limits overfitting, though single-tree splits trail the ensemble in precision (**0.4410**) and AUC[., 14]. |
| kNN | Records the highest **Accuracy (0.9051)** and highest **Precision (0.6148)**, but has low minority-class **Recall (0.4213)**, limiting **F1 (0.5000)** and **MCC (0.4594)**[., 14]. Without native class-weight balancing and with distance computations across one-hot encoded features, predictions skew toward the 88.73% majority class (`0`)[., 14]. |
| Naive Bayes | Shows the lowest overall scores across **Accuracy (0.7217)**, **AUC (0.8470)**, **Precision (0.2681)**, **F1 (0.4076)**, and **MCC (0.3665)** despite **Recall = 0.8502**[., 14]. Its conditional independence and Gaussian assumptions do not hold well for correlated macroeconomic series (`euribor3m`, `nr.employed`, `emp.var.rate`) and sparse one-hot columns, producing many false positives[.]. |
| Random Forest (Ensemble) | Delivers the best overall balance on imbalanced data, ranking first in **AUC (0.9503)**, **F1 (0.6355)**, and **MCC (0.6110)** alongside **Recall = 0.9019** and **Accuracy = 0.8835**[., 14]. Ensemble averaging across 200 trees (`max_depth=14`, `class_weight="balanced"`) captures non-linear feature interactions while reducing variance[.]. |
| Overall Winner for your dataset? | **Random Forest (Ensemble)** is the overall winner because it achieves the highest **MCC (0.6110)**, **F1 Score (0.6355)**, and **AUC (0.9503)** under class imbalance while maintaining high recall (**0.9019**)[., 14]. |