import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

st.set_page_config(page_title="Credit Risk Prediction", layout="wide")

# Title
st.title("📊 Credit Risk Classification App")
st.markdown("This app loads and trains a model to predict good vs bad credit risk using XGBoost.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("german_credit_data.csv")
    return df

credit_data = load_data()

# Preprocessing
credit_data['Checking account'] = pd.to_numeric(credit_data['Checking account'], errors='coerce')
credit_data['Checking account'].fillna(credit_data['Checking account'].median(), inplace=True)
credit_data['Saving accounts'].fillna('unknown', inplace=True)

# Thresholds
credit_limit = credit_data['Credit amount'].median()
time_limit   = credit_data['Duration'].median()

credit_data['Target'] = (
    (credit_data['Credit amount'] > credit_limit) &
    (credit_data['Duration'] < time_limit)
).astype(int)

# Encoding
encoders = {}
for col in credit_data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    credit_data[col] = le.fit_transform(credit_data[col])
    encoders[col] = le

# Scaling
features_to_scale = ['Age', 'Credit amount', 'Duration', 'Checking account']
scaler = StandardScaler()
credit_data[features_to_scale] = scaler.fit_transform(credit_data[features_to_scale])

# Show dataframe
with st.expander("🔍 View Preprocessed Dataset"):
    st.dataframe(credit_data.head())

# Heatmap
st.subheader("🔗 Feature Correlation Heatmap")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(credit_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
st.pyplot(fig)

# Train/Test split
X = credit_data.drop(columns=['Target'])
y = credit_data['Target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Hyperparameters
st.sidebar.header("⚙️ Model Parameters")
n_estimators = st.sidebar.selectbox("n_estimators", [100, 200], index=0)
max_depth = st.sidebar.selectbox("max_depth", [3, 5, 7], index=1)
learning_rate = st.sidebar.selectbox("learning_rate", [0.01, 0.1, 0.2], index=1)

# Train Model
st.subheader("📈 Model Training and Evaluation")
if st.button("Train Model"):
    with st.spinner("Training the XGBoost model..."):
        model = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    st.success("Model training completed!")

    # Evaluation
    st.markdown("#### Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # Confusion Matrix
    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
                xticklabels=['Good', 'Bad'], yticklabels=['Good', 'Bad'], ax=ax2)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig2)
else:
    st.info("Click the 'Train Model' button to run the model.")

