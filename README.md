# 🍷 Wine Quality Prediction Pipeline for Imbalanced Multiclass Classification

### A Gradient Boosting Approach with OOF Evaluation


---

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-MachineLearning-orange)
![Imbalanced-Learn](https://img.shields.io/badge/SMOTE-Imbalanced-green)
![Status](https://img.shields.io/badge/Project-Academic-success)

---

## 📌 Overview

This project develops an end-to-end machine learning pipeline for multiclass wine quality prediction based on physicochemical characteristics.

The dataset presents two major challenges:

- imbalanced class distribution,
- and strong overlap between adjacent wine quality classes.

Most observations belong to quality **5** and **6**, while classes such as **3** and **8** contain only a small number of samples. In addition, neighboring quality classes often exhibit very similar physicochemical properties, making class boundaries difficult to separate.

To address these issues, the pipeline incorporates:

- feature engineering based on physicochemical ratios,
- feature scaling using `StandardScaler`,
- SMOTE within cross-validation pipelines to avoid data leakage,
- ensemble-based model comparison,
- and Out-of-Fold (OOF) evaluation for unbiased performance estimation.

The final selected model uses **Gradient Boosting** due to its superior ability to handle overlapping decision boundaries across wine quality classes.

---

## ⚙️ Workflow

```text
Load Data
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Feature Scaling
   ↓
SMOTE Pipeline
   ↓
Model Comparison
   ↓
Model Selection
   ↓
OOF Evaluation
   ↓
Test Prediction
   ↓
Submission File Generation
````

---

## 📂 Repository Structure

```text
wine-quality-classification-pipeline/
│
├── data_training.csv
├── data_testing.csv
├── Wine_Quality_Pipeline.ipynb
├── hasilprediksi_023.csv
├── README.md
└── requirements.txt
```

---

## 📊 Exploratory Data Analysis

Several important findings were obtained during EDA:

* Wine quality labels are heavily imbalanced.
* Features such as `residual sugar` and `total sulfur dioxide`
  have very large value ranges.
* Several variables exhibit skewed distributions and outliers.
* Adjacent classes (5, 6, and 7) show strong overlap in feature space.

These findings strongly influenced the preprocessing and evaluation strategy used throughout the pipeline.

---

## 🧪 Feature Engineering

Three additional features were created using domain-inspired physicochemical ratios:

| Feature                 | Description                                                |
| ----------------------- | ---------------------------------------------------------- |
| `acidity_ratio`         | Balance between fixed acidity and volatile acidity         |
| `sulfur_ratio`          | Proportion of active sulfur dioxide                        |
| `alcohol_density_ratio` | Interaction between alcohol concentration and wine density |

The engineered features significantly contributed to the final model performance and appeared among the highest feature importance scores.

---

## ⚖️ Handling Imbalanced Data

Class imbalance was handled using **SMOTE** inside an `ImbPipeline`.

This approach prevents synthetic samples from leaking into validation folds during cross-validation.

```python
pipeline_smote = ImbPipeline([
    ('smote', SMOTE(random_state=42, k_neighbors=3)),
    ('clf', RandomForestClassifier(...))
])
```

Placing SMOTE outside the CV pipeline may produce overly optimistic validation results due to hidden data leakage.

---

## 🤖 Model Comparison

The following models were evaluated using Stratified K-Fold Cross Validation:

| Model                 | Description                             |
| --------------------- | --------------------------------------- |
| Random Forest         | Ensemble bagging approach               |
| Random Forest + SMOTE | RF combined with synthetic oversampling |
| Gradient Boosting     | Sequential boosting-based ensemble      |

Gradient Boosting achieved the best overall validation performance.

---

## 📈 Evaluation Strategy

Evaluation was performed using:

* Cross Validation Accuracy
* OOF Accuracy
* Macro F1-Score
* Weighted F1-Score
* Classification Report
* Confusion Matrix

OOF prediction was used to obtain a more realistic estimate of generalization performance.

Unlike training accuracy, OOF evaluation ensures that every observation is predicted by a model that never saw that observation during training.

---

## 📌 Key Findings

* The primary challenge was not only class imbalance,
  but also overlap between neighboring wine quality classes.
* SMOTE did not significantly improve performance,
  indicating that boundary ambiguity was more dominant than sample scarcity.
* Gradient Boosting outperformed Random Forest
  due to its ability to iteratively focus on difficult observations.
* Feature engineering contributed meaningful predictive information.

---

## 📊 Final Results

| Metric            | Score |
| ----------------- | ----- |
| OOF Accuracy      | ~0.63 |
| Macro F1-Score    | ~0.36 |
| Weighted F1-Score | ~0.62 |

The relatively lower Macro F1-Score indicates that minority classes remain difficult to classify consistently.

However, the model still demonstrates reasonable capability in recognizing minority patterns while maintaining stable overall performance.

---

## 🛠️ Libraries Used

* pandas
* numpy
* scikit-learn
* imbalanced-learn
* matplotlib
* seaborn

---

## ▶️ How to Run

Clone this repository:

```bash
git clone https://github.com/username/wine-quality-classification-pipeline.git
cd wine-quality-classification-pipeline
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook Wine_Quality_Pipeline.ipynb
```

---

## 📚 Academic Context

This project was developed as part of:

**Penambangan Data dan Analisis Bisnis**
S1 Statistika dan Sains Data — Universitas Negeri Semarang (UNNES)

Semester Genap 2025/2026.

---

## 👤 Author

M Irfan Al Hakim

---
