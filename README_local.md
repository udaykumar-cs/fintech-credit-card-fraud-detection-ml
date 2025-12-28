# 💳 Credit Card Fraud Detection using Advanced Machine Learning Pipelines

## 📚 Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Dimensionality Reduction & Visualization](#dimensionality-reduction--visualization)
- [Models Implemented](#models-implemented)
  - [Isolation Forest](#isolation-forest-unsupervised)
  - [Logistic Regression + SMOTE](#logistic-regression--smote-pipeline)
  - [Random Forest Classifier](#random-forest-classifier)
  - [XGBoost](#xgboost-final-model)
- [Machine Learning Pipeline Design](#machine-learning-pipeline-design)
- [System Architecture Diagram](#system-architecture-diagram)
- [Threshold Optimization](#threshold-optimization)
- [Model Evaluation Metrics](#model-evaluation-metrics)
- [Final Results](#final-results-xgboost--threshold--029)
- [Model Comparison & Visualization](#model-comparison--visualization)
- [Business Impact](#business-impact) 
- [Tech Stack](#tech-stack)
- [Deployment Readiness](#deployment-readiness)
- [How to Run](#how-to-run-the-project)
- [Limitations](#limitations)
- [License](#license)
- [Author](#author)


## 🔍 Project Overview


This project implements an **end-to-end credit card fraud detection system** using **exploratory data analysis, dimensionality reduction, anomaly detection, supervised learning pipelines, ensemble models, and probability threshold optimization**.

The dataset is **extremely imbalanced (<0.2% fraud cases)**, so the project focuses on **recall-driven evaluation**, avoiding the accuracy paradox common in fraud detection problems.

---

## 📌 Problem Statement
  
Detect fraudulent credit card transactions while:

* Minimizing **false negatives** (missed frauds)
* Controlling **false positives** (customer inconvenience)
* Handling severe **class imbalance**

---

## 📊 Dataset Description

* **Source**: Credit Card Transactions Dataset
* **Target Column**: `Class`

  * `0` → Legitimate transaction
  * `1` → Fraudulent transaction
* **Features**:

  * Fully numerical (PCA-transformed for privacy)
  * Includes `Amount` and `Time`
* **Class Distribution**:

  * Legitimate ≈ 99.8%
  * Fraud ≈ 0.2%

---

## 🔎 Exploratory Data Analysis (EDA)
  
The following steps were performed:

* Dataset shape, info, and statistical summary
* Missing value and duplicate analysis
* Removal of duplicate rows
* Target imbalance visualization (pie chart)
* Correlation heatmap for top features
* Outlier analysis using boxplot on `Amount`

**Insight:** The dataset is clean but extremely skewed, making accuracy an unreliable metric.

---

## 📉 Dimensionality Reduction & Visualization
 
### PCA (Principal Component Analysis)
 
* Linear dimensionality reduction technique
* Preserves maximum variance
* Used for **2D visualization of fraud vs non-fraud**
* Also reused for visualizing Isolation Forest anomalies

### t-SNE (t-distributed Stochastic Neighbor Embedding)
 
* Non-linear, neighborhood-preserving projection
* Applied on a **sample of 8,000 points** due to computational cost
* Used **only for exploratory visualization**, not for modeling

⚠️ *t-SNE is non-deterministic and unsuitable for training pipelines*

---

## 🧪 Models Implemented

### 1️⃣ Isolation Forest (Unsupervised)
 
* Learns normal transaction patterns
* Flags anomalies using random partitioning
* Useful for **early-stage anomaly screening**

**Observation:**

* High false positives
* Low fraud precision
* Not suitable as a standalone classifier

---

### 2️⃣ Logistic Regression + SMOTE (Pipeline)
 
* Linear baseline model
* Implemented using `imblearn.Pipeline`

```text
[ StandardScaler → SMOTE → LogisticRegression ]
```

**Why this model?**

* Establishes a baseline
* Demonstrates effect of oversampling
* High recall but very low precision

---

### 3️⃣ Random Forest Classifier

* Bagging-based ensemble of decision trees
* Captures non-linear feature interactions
* Robust to noise and outliers

**Why Random Forest?**

* Strong traditional ML benchmark
* Comparison point for boosting models

**Observation:**

* Excellent AUC-ROC
* Slightly lower fraud recall than XGBoost

---

### 4️⃣ XGBoost (Final Model)

* Gradient Boosted Decision Trees
* Optimizes errors sequentially using gradient descent
* Strong regularization controls overfitting

**Key Hyperparameters**:

* `n_estimators=300`
* `max_depth=8`
* `learning_rate=0.05`
* `subsample=0.8`
* `gamma=0.1`

---

## ⚙️ Machine Learning Pipeline Design

All supervised models use **pipelines** to ensure:

* No data leakage
* Correct SMOTE application (train-only)
* Reproducibility and deployment readiness

---

## 🧠 Key Design Decisions

* Used pipelines to prevent data leakage and ensure correct SMOTE usage
* Avoided accuracy due to extreme class imbalance
* Optimized probability threshold instead of model parameters alone
* Treated anomaly detection as a support tool, not a final classifier
* Preferred recall-sensitive evaluation aligned with fraud detection costs

---

## 🏗️ System Architecture Diagram
 
The following diagram illustrates the **end-to-end workflow** of the fraud detection system, from raw data ingestion to final fraud decision. It highlights preprocessing, model pipelines, ensemble comparison, and threshold optimization.

```text
+---------------------------+
|   Credit Card Dataset     |
| (Raw Transactions Data)   |
+-------------+-------------+
              |
              v
+---------------------------+
|  Data Cleaning & EDA      |
| - Remove duplicates       |
| - Analyze imbalance       |
| - Correlation analysis    |
+-------------+-------------+
              |
              v
+---------------------------+
| Feature Scaling           |
| (StandardScaler)          |
+-------------+-------------+
              |
              v
+---------------------------+
| Exploratory Visualization |
| - PCA (2D Projection)     |
| - t-SNE (Sampled Data)    |
+-------------+-------------+
              |
              v
+---------------------------+
| Train-Test Split          |
| (Stratified)              |
+-------------+-------------+
              |
              v
+------------------------------------------------+
| Machine Learning Pipelines                     |
|                                                |
|  +----------------------+                      |
|  | SMOTE (Train Only)   |                      |
|  +----------+-----------+                      |
|             |                                  |
|   +---------v---------+   +----------------+  |
|   | Logistic Regression |  | Random Forest |  |
|   +--------------------+   +----------------+  |
|                                                |
|            +----------------------+            |
|            |      XGBoost         |            |
|            |  (Final Model)       |            |
|            +----------+-----------+            |
+---------------------------+--------------------+
                            |
                            v
+---------------------------+
| Probability Prediction    |
| (Fraud Probability)       |
+-------------+-------------+
              |
              v
+---------------------------+
| Threshold Optimization    |
| (Best Threshold = 0.29)   |
+-------------+-------------+
              |
              v
+---------------------------+
| Final Fraud Decision      |
| (Fraud / Not Fraud)       |
+---------------------------+

```
## 🎯 Threshold Optimization

Instead of using the default threshold (0.5), **F1-score based threshold tuning** was applied.

```python
Best Threshold = 0.29
```

**Why?**

* Fraud detection is recall-sensitive
* Probability calibration enables business-aligned decisions

---

## 📊 Model Evaluation Metrics

Accuracy is avoided as a primary metric.

| Metric    | Reason                                |
| --------- | ------------------------------------- |
| Precision | Measures false alarm cost             |
| Recall    | Measures missed fraud cost            |
| F1-Score  | Balance between precision & recall    |
| AUC-ROC   | Threshold-independent ranking ability |

---

## 🏆 Final Results (XGBoost @ Threshold = 0.29)
 
* Accuracy: **~99.95%**
* Fraud Recall: **~77%**
* Precision: **~97%**
* AUC-ROC: **~0.97**

This configuration achieves a business-optimal balance between fraud detection recall and customer impact due to false positives.

---

## 📈 Model Comparison & Visualization

* ROC Curve comparison (LogReg vs RF vs XGBoost)
* Precision–Recall curves
* AUC comparison bar chart
* Final performance summary table

---

## 💼 Business Impact

- Reduces direct financial losses by improving fraud recall
- Minimizes customer friction by controlling false positives
- Demonstrates how threshold tuning aligns ML decisions with business risk
- Suitable for deployment in high-volume FinTech transaction systems


## 💾 Model Persistence

* Final trained pipeline saved using `joblib`
* Ready for inference and deployment

---

## 🛠 Tech Stack

* Python
* NumPy, Pandas
* Scikit-learn
* XGBoost
* Imbalanced-learn (SMOTE)
* Matplotlib, Seaborn

---

## 🧩 Deployment Readiness

- Trained ML pipeline saved as a **single serialized artifact**
- **StandardScaler, SMOTE, and model** bundled together to avoid data leakage
- Can be directly loaded for:
  - Batch inference
  - REST API (FastAPI / Flask)
  - Real-time fraud scoring systems


## 🚀 Future Enhancements
 
* Cost-sensitive learning
* SHAP explainability
* Real-time streaming fraud detection
* Automated hyperparameter tuning

---

## ▶️ How to Run the Project
  
 1. Clone the repository
2. Install dependencies:
   pip install -r requirements.txt
3. Run notebooks in order:
   - 1_Data_Loading_and_Exploration.ipynb
   - 2_Modeling_and_Evaluation.ipynb
4. Final model is saved using joblib for inference

---

## ⚠️ Limitations
  
 -Dataset is anonymized and PCA-transformed, limiting feature interpretability
- Threshold tuning is dataset-specific and may require recalibration in production
- No real-time latency constraints were evaluated
- Concept drift handling is not implemented

## 📄 License

This project is licensed under the **MIT License**.


## 👨‍💻 Author
 
**Uday Kumar**
B.Tech – Data Science
Focused on Applied Machine Learning & AI Systems
https://github.com/udaykumar-cs
https://www.linkedin.com/in/uday-kumar-uk/
📧 kumaruday9973@gmail.com
---

⭐ If you find this project useful, consider starring the repository!
