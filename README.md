# Diabetes Classification: Gaussian Naive Bayes
**Project:** Medical Diagnosis Classification Task

## 1. Project Overview
This project builds a machine learning pipeline to predict whether a patient has diabetes based on diagnostic measurements. The goal was to demonstrate the effectiveness of probabilistic modeling on biological data using the **Gaussian Naive Bayes** algorithm.

* **Dataset:** Kaggle Diabetes Classification (995 entries).
* **Features:** `glucose` (Blood Glucose Level), `bloodpressure` (Blood Pressure).
* **Target:** `diabetes` (0 = Negative, 1 = Positive).
* **Tech Stack:** Python, Scikit-learn (GaussianNB), Matplotlib, Seaborn.

---

## 2. Visual Analysis & Feature Distributions
I performed a visual analysis to understand the underlying statistical properties of the data, which justified the choice of model.

### Likelihood Distributions
I plotted the Gaussian probability density functions for `glucose` and `bloodpressure`. The separation between the "Diabetic" and "Not Diabetic" curves indicated that these features are strong predictors.

<img width="1384" height="583" alt="LIKELIHOOD" src="https://github.com/user-attachments/assets/15e5e7f5-fbb9-498e-bb4b-26fc7698afc7" />

### Decision Boundary
Unlike linear models, the Naive Bayes classifier produced a smooth, elliptical decision boundary that effectively encapsulated the diabetic class within the high-glucose/high-blood-pressure region.

<img width="854" height="635" alt="DECISION" src="https://github.com/user-attachments/assets/9f9860f0-01f1-46c4-a09c-33e262d96536" />

* **Insight:** The visual analysis confirmed that the data classes form distinct clusters that can be separated probabilistically.

---

## 3. Predictive Modeling
I implemented a supervised learning workflow focused on probabilistic classification.

### Data Preparation
* **Splitting:** 75/25 Train-Test split.
* **Scaling:** Applied `StandardScaler` to normalize glucose and blood pressure values. While Naive Bayes is not strictly dependent on scaling, this step ensures numerical stability and makes visualization (like the decision boundary) interpretable.

### Model Selection
* **Model:** Gaussian Naive Bayes (`GaussianNB`).
* **Rationale:** The features are continuous and follow a bell-curve (normal) distribution, making Gaussian NB the ideal assumption-based model.
* **Efficiency:** The model is computationally efficient and requires very little training data to estimate parameters ($\mu$ and $\sigma$).

<img width="790" height="635" alt="ROC" src="https://github.com/user-attachments/assets/7549b631-1144-442e-84b7-e2529721aa32" />

---

## 4. Performance & Results
The model demonstrated high reliability with balanced performance across both classes.

### Evaluation Metrics
* **Accuracy:** **92.77%**
* **AUC-ROC Score:** **0.9805** (Indicates excellent separability between classes).
* **F1-Score:** 0.93

### Confusion Matrix
The model correctly identified the vast majority of cases with minimal false positives/negatives.

<img width="663" height="558" alt="confusion" src="https://github.com/user-attachments/assets/0e13c12c-d64a-4ede-92cb-784ef4338050" />

* **True Negatives:** 105
* **True Positives:** 126
* **False Negatives:** 9 (Critical error in medical diagnosis, but the count is low here).

### Conclusion
The visual analysis confirmed that the data adhered well to the Gaussian assumption. Consequently, the **Gaussian Naive Bayes** classifier proved to be a robust and highly accurate solution for this biological dataset.
