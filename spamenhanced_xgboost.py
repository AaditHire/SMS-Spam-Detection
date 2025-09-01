import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

st.title("🚀 SMS Spam Detection ")
st.write("Detect spam messages and experiment with different XGBoost hyperparameters .")

uploaded_file = r"C:\Users\spam.csv"

try:
    df = pd.read_csv(uploaded_file, encoding='latin1')
    
    st.success("Dataset loaded successfully!")
    st.write("First few rows of your dataset:")
    st.dataframe(df.head())

    if 'label' in df.columns and 'message' in df.columns:
        clean_df = df[['label', 'message']]
    else:
        clean_df = df[['v1', 'v2']]
        clean_df.columns = ['label', 'message']

    st.subheader("Message Distribution")
    msg_count = clean_df['label'].value_counts()
    st.bar_chart(msg_count)

    X = clean_df['message']
    y = clean_df['label'].map({'ham': 0, 'spam': 1}) 

    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)

    st.sidebar.header("🎛️ Hyperparameter Tuning")
    n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 50, 500, 100, step=10)
    max_depth = st.sidebar.slider("Maximum Tree Depth (max_depth)", 1, 20, 6)
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.3, step=0.01)

    st.sidebar.write(f"**n_estimators:** {n_estimators}")
    st.sidebar.write(f"**max_depth:** {max_depth}")
    st.sidebar.write(f"**learning_rate:** {learning_rate}")

    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.write(f"Model Accuracy: **{acc * 100:.2f}%**")

    with st.expander("See Classification Report"):
        report = classification_report(y_test, y_pred, target_names=['ham', 'spam'], output_dict=False)
        st.text(report)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)

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

    st.header("Test your messages")
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

