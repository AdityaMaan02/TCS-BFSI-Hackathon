import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib
import os

# 1. Load dataset
credit_data = pd.read_csv("german_credit_data.csv")

# 2. Clean 'Checking account' as numeric, impute median
credit_data['Checking account'] = pd.to_numeric(credit_data['Checking account'], errors='coerce')
credit_data['Checking account'].fillna(credit_data['Checking account'].median(), inplace=True)

# 3. Handle missing values in 'Saving accounts'
credit_data['Saving accounts'].fillna('unknown', inplace=True)

# 4. Compute thresholds on RAW data (before scaling)
credit_limit = credit_data['Credit amount'].median()
time_limit   = credit_data['Duration'].median()

# 5. Define target
credit_data['Target'] = (
    (credit_data['Credit amount'] > credit_limit) &
    (credit_data['Duration']     < time_limit)
).astype(int)

# 6. Encode categorical features (only object dtype now)
encoders = {}
for col in credit_data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    credit_data[col] = le.fit_transform(credit_data[col])
    encoders[col] = le

# 7. Scale numerical features (including Checking account)
features_to_scale = ['Age', 'Credit amount', 'Duration', 'Checking account']
scaler = StandardScaler()
credit_data[features_to_scale] = scaler.fit_transform(credit_data[features_to_scale])

# 8. Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(credit_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# 9. Split and tune
X = credit_data.drop(columns=['Target'])
y = credit_data['Target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

param_grid = {
    'n_estimators':   [100, 200],
    'max_depth':      [3, 5, 7],
    'learning_rate':  [0.01, 0.1, 0.2],
}

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
grid = GridSearchCV(model, param_grid, scoring='f1', cv=3, verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)

best = grid.best_estimator_
y_pred = best.predict(X_test)

# 10. Results
print("Best params:", grid.best_params_)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
            xticklabels=['Good','Bad'], yticklabels=['Good','Bad'])
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()