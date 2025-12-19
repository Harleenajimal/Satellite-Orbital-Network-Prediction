# Satellite-Orbital-Network-Prediction
Satellite Orbital Congestion Risk Prediction using Machine Learning

## 📌 Project Overview

With the rapid increase in active satellites and space debris, orbital congestion has become a serious concern for space safety. This project focuses on analyzing satellite orbital parameters and predicting **orbital congestion / collision risk levels** using Machine Learning.

Instead of relying on a pre-defined target variable, this project **engineers a custom risk score** based on orbital mechanics and then evaluates multiple ML models to classify satellites into **Low, Medium, and High risk categories**.

---

## 🎯 Objectives

### Objective 1 (Exploratory & Feature Understanding)

* Understand key orbital parameters affecting congestion
* Clean and preprocess real-world satellite orbital data

### Objective 2 (Core ML Objective)

* Engineer a domain-based **Orbital Risk Score**
* Convert the risk score into categorical risk levels
* Train and compare multiple ML models
* Identify the most influential orbital features

---

## 📂 Dataset

* **Source:** Kaggle – Satellite Orbital Catalog (CelesTrak-based)
* **Type:** Real-world satellite orbital data

### Key Features Used

* `MEAN_MOTION` – Indicates orbital speed / altitude (lower altitude → higher congestion)
* `INCLINATION` – Determines orbital plane overlap
* `ECCENTRICITY` – Measures orbital stability

These features were selected based on their relevance to orbital congestion and collision probability.

---

## 🧠 Feature Engineering: Orbital Risk Score

Since no direct risk label was provided, a **custom orbital risk score** was designed:

* Higher mean motion → crowded lower orbits
* Similar inclinations → higher collision chance
* Low eccentricity → stable but congested orbits

The score was computed using normalized orbital parameters and weighted based on domain reasoning.

The continuous risk score was then divided into three categories:

* **Low Risk**
* **Medium Risk**
* **High Risk**

---

## 🤖 Machine Learning Models Used

Multiple models were trained and evaluated to compare performance:

* Logistic Regression (baseline)
* Random Forest Classifier
* Gradient Boosting Classifier

### Evaluation Metrics

* Accuracy
* Precision, Recall, F1-score
* Confusion Matrix
* Feature Importance (for tree-based models)

---

## 📊 Results & Insights

* Ensemble models (Random Forest & Gradient Boosting) outperformed linear models
* Mean Motion and Inclination were identified as the strongest contributors to orbital risk
* The model successfully categorized satellites into meaningful risk levels

---

## 🔍 Key Takeaways

* Demonstrates **end-to-end ML workflow** on a non-trivial dataset
* Shows ability to define a problem when labels are not provided
* Highlights real-world data challenges such as missing features and schema inconsistencies
* Combines domain understanding with machine learning techniques

---

## 🚀 Future Improvements

* Add unsupervised clustering (DBSCAN) to detect congested orbital regions
* Convert risk prediction into a regression-based approach
* Integrate advanced models like XGBoost
* Build a visualization or dashboard for orbital risk monitoring

---

## 🛠 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Jupyter Notebook

---

## 📌 Conclusion

This project presents a practical and innovative approach to satellite orbital risk analysis using Machine Learning. By engineering a domain-driven risk score and evaluating multiple models, the project goes beyond standard datasets and demonstrates strong analytical and problem-solving skills.
