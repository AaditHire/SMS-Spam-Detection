# 📊 SMS Spam Detection with Streamlit and XGBoost

This project is a simple yet powerful web application for detecting spam SMS messages using machine learning. The app is built with **Streamlit** and uses the **XGBoost** classifier for accurate spam classification. It also provides useful performance visualizations like Confusion Matrix and ROC Curve.

---

## 🚀 Features

- Detects spam or ham (non-spam) messages
- Built-in **XGBoost** model for high accuracy
- Interactive **hyperparameter tuning** via sidebar
- Real-time predictions for custom messages
- Visualizations:
  - Class distribution chart
  - Confusion matrix heatmap
  - ROC Curve with AUC score
- User-friendly Streamlit interface

---

## 🛠️ Tech Stack

- Python 3.x
- Streamlit
- Scikit-learn
- XGBoost
- Pandas
- Matplotlib & Seaborn

---

## 📁 Dataset

The project uses the popular **SMS Spam Collection Dataset**, consisting of labeled messages as 'ham' (not spam) or 'spam'. You can download the dataset from:

🔗 [UCI Repository - SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

**Note:** In the provided code, the dataset is loaded from:

```plaintext
C:\Users\ASUS\Downloads\archive\spam.csv
