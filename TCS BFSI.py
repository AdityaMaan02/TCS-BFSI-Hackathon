import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Page configuration
st.set_page_config(page_title="Credit Risk Predictor", layout="wide")

# Load and preprocess data
def load_data():
    df = pd.read_csv("german_credit_data.csv")
    df['Class'] = df['Credit amount'].apply(lambda x: 1 if x > 5000 else 0)
    return df

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Title
st.title("💳 Credit Risk Prediction App")
st.markdown("Use this app to analyze and predict credit risk based on customer data from the German Credit dataset.")

# Sidebar
with st.sidebar:
    st.header("📂 Data Source")
    uploaded_file = st.file_uploader("Upload your dataset", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Custom dataset loaded")
    else:
        df = load_data()
        st.info("Using default dataset")

# Main Section
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# Prepare data
X = df.drop(columns=['Class', 'Unnamed: 0'], errors='ignore')
y = df['Class']
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training
model = train_model(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
st.metric("🔍 Model Accuracy", f"{accuracy:.2%}")

# Tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["📋 Classification Report", "📉 Confusion Matrix", "🔥 Feature Importance"])

with tab1:
    st.code(classification_report(y_test, y_pred), language='text')

with tab2:
    st.write("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=['Good', 'Bad'], yticklabels=['Good', 'Bad'])
    st.pyplot(fig)

with tab3:
    st.write("Top 10 Important Features")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.barplot(data=feat_df.head(10), x="Importance", y="Feature", palette="mako", ax=ax2)
    st.pyplot(fig2)

# Custom Prediction Input
st.subheader("🔮 Predict Credit Risk for Custom Input")
with st.form("prediction_form"):
    cols = st.columns(2)
    sample_input = []
    for i, feature in enumerate(X.columns):
        with cols[i % 2]:
            value = st.number_input(f"{feature}", value=0.0)
            sample_input.append(value)
    submitted = st.form_submit_button("Predict Risk")

if submitted:
    prediction = model.predict(np.array(sample_input).reshape(1, -1))
    result = "🟢 Good Credit" if prediction[0] == 0 else "🔴 Bad Credit"
    st.success(f"Prediction Result: {result}")
