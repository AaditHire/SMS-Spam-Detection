import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

st.title("🚀 SMS Spam Detection with XGBoost and Visual Insights")
st.write("Detect spam messages and visualize model performance using the powerful XGBoost algorithm.")

# Load your dataset
uploaded_file = r"C:\Users\ASUS\Downloads\archive\spam.csv"

try:
    df = pd.read_csv(uploaded_file, encoding='latin1')
    
    st.success("Dataset loaded successfully!")
    st.write("First few rows of your dataset:")
    st.dataframe(df.head())

    # Cleaning dataset
    if 'label' in df.columns and 'message' in df.columns:
        clean_df = df[['label', 'message']]
    else:
        clean_df = df[['v1', 'v2']]
        clean_df.columns = ['label', 'message']

    # Show class distribution
    st.subheader("Message Distribution")
    msg_count = clean_df['label'].value_counts()
    st.bar_chart(msg_count)

    # Preprocessing
    X = clean_df['message']
    y = clean_df['label'].map({'ham': 0, 'spam': 1})  # Encode labels as 0 (ham) and 1 (spam)

    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)

    # XGBoost Model
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.write(f"Model Accuracy: **{acc * 100:.2f}%**")

    with st.expander("See Classification Report"):
        report = classification_report(y_test, y_pred, target_names=['ham', 'spam'], output_dict=False)
        st.text(report)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)

    # ROC Curve
    y_scores = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    st.subheader("ROC Curve")
    fig2, ax2 = plt.subplots()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend()
    st.pyplot(fig2)

    # User Input
    st.header("Try it Yourself")
    user_input = st.text_area("Enter your message:")

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter a message.")
        else:
            input_vector = vectorizer.transform([user_input])
            prediction = model.predict(input_vector)[0]
            
            if prediction == 1:
                st.error("⚠️ This message is likely **SPAM**.")
            else:
                st.success("✅ This message is **NOT Spam**.")

except Exception as e:
    st.error(f"Error loading dataset: {e}")
