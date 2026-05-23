# 🫀 HEART SCAN v1.0: Ischemic Heart Disease Classifier

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]
([(https://heart-classifier-ds7hjowzcugfv2ul8evyza.streamlit.app/)])

**[Brief 1-2 sentence pitch, e.g., "A machine learning pipeline and interactive web application designed to evaluate clinical biometric data and predict the probability of coronary artery disease."]**

![App Screenshot]
([(image.png)])
*HEART SCAN v1.0 - Cyber-Luxury Interface*

## 🌐 Live Application
The model is deployed and fully accessible via Streamlit Community Cloud:
👉 **[Access the Live Neural Scan Here]
([https://heart-classifier-ds7hjowzcugfv2ul8evyza.streamlit.app/])**

---

## 🛠️ Tech Stack & Methodology

This project is broken into two distinct phases: model training and production inference. 

* **Frontend Framework:** Streamlit
* **Machine Learning:** scikit-learn (Logistic Regression)
* **Data Manipulation:** pandas, NumPy
* **Deployment:** Streamlit Community Cloud

### The Pipeline
1. **Data Preprocessing:** Imputed missing values (median for numerical, mode for categorical) and one-hot encoded categorical variables.
2. **Model Training:** Utilized Logistic Regression, scaled via `StandardScaler`, and evaluated using accuracy, recall, precision, and confusion matrices.
3. **Inference UI:** A decoupled, lightweight Streamlit frontend that loads pre-trained `.pkl` artifacts for rapid, zero-lag predictions.

---

## 📁 Repository Structure

```text
├── app.py                     # Streamlit frontend (Inference only)
├── heart_disease_model.pkl    # Trained Logistic Regression model
├── scaler.pkl                 # StandardScaler fitted to training data
├── feature_names.pkl          # Saved feature columns to align user input
├── requirements.txt           # Production dependencies for Streamlit Cloud
└── heart_disease_uci.csv      # Original dataset (used during training phase)