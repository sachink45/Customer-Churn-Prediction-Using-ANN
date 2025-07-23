# 🧠 Customer Churn Prediction Using ANN

This project predicts customer churn using an **Artificial Neural Network (ANN)** built with **TensorFlow/Keras** and provides an interactive **Streamlit web application** for real-time predictions.

---

## 🔍 Overview

Customer churn occurs when customers stop using a company's services. This project helps businesses identify potential churners so they can take preventive actions.

The model is trained on a telecom customer dataset, and the Streamlit app allows users to input customer data and get a churn prediction instantly.

---

## ✅ Features

- Deep learning-based churn prediction using ANN
- Cleaned and preprocessed dataset
- Streamlit web UI for easy interaction
- Real-time prediction with user input
- Well-organized and readable code

---

## 🧠 Model (ANN)

- **Framework:** TensorFlow / Keras
- **Architecture:**
  - Input Layer: Customer features
  - 2 Hidden Layers: Dense layers with ReLU activation
  - Output Layer: Sigmoid activation for binary classification
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam
- **Metric:** Accuracy

---

## 🚀 Streamlit Web App

The `stream.py` file runs a Streamlit application where you can input customer details (e.g., gender, contract type, monthly charges, etc.) and receive a prediction: **Will this customer churn?**

### To run the app locally:

```bash
git clone https://github.com/sachink45/Customer-Churn-Prediction-Using-ANN.git

cd Customer-Churn-Prediction-Using-ANN

python -m venv venv

venv\Scripts\activate   

pip install -r requirements.txt

streamlit run app.py



