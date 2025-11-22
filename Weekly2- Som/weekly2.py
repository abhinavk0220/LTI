# high_value_purchase_pipeline.py
# Predicting High-Value Purchases (binary classification)
# -------------------------
# PIP INSTALLS (run these in your environment / terminal before running the script):
# pip install pandas numpy scikit-learn matplotlib seaborn tensorflow==2.12.0
# (If you prefer a later TF and your environment supports it, you can omit the version pin)

"""
Complete end-to-end script that:
1) Loads two CSVs: purchase_data.csv and user_demo.csv
2) Merges, cleans, encodes, scales
3) Trains a baseline Logistic Regression
4) Trains a simple Keras MLP (binary classifier)
5) Evaluates and compares both

Notes:
- Update file paths for your CSVs if needed.
- This script uses reasonable defaults for missing-value handling and encoding.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# -------------------------
# 1) LOAD DATA
# -------------------------
# Replace paths below if files are elsewhere
purchase_path = "User_product_purchase_details_p2.csv"
user_path = "user_demographics.csv"

if not os.path.exists(purchase_path) or not os.path.exists(user_path):
    raise FileNotFoundError("Make sure 'purchase_data.csv' and 'user_demo.csv' exist in the working directory.")

df_purchase = pd.read_csv(purchase_path)
df_user = pd.read_csv(user_path)

print("purchase rows:", df_purchase.shape)
print("user rows:", df_user.shape)

# -------------------------
# 2) MERGE
# -------------------------
df = pd.merge(df_purchase, df_user, on="User_ID", how="left")
print("merged shape:", df.shape)

# -------------------------
# 3) TARGET and DROPS
# -------------------------
# Create binary target
df['High_Value_Purchase'] = (df['Purchase'] >= 10000).astype(int)

# Drop columns not needed for modeling (keep User_ID if you want to group/aggregate later)
cols_to_drop = ['Product_ID']  # user can add more if needed
for c in cols_to_drop:
    if c in df.columns:
        df = df.drop(columns=[c])

# -------------------------
# 4) MISSING VALUE HANDLING
# -------------------------
# Inspect common missing columns: Product_Category_2/3 or user fields
# Strategy used here:
# - For categorical product categories: fill with 0 (or -1) to indicate missing
# - For City_Category / Stay_In_Current_City_Years: fill with mode
# - For numerical fields (if any): fill with median

# Product categories often have NaNs
for c in ['Product_Category_2', 'Product_Category_3']:
    if c in df.columns:
        df[c] = df[c].fillna(0).astype(int)

# If Stay_In_Current_City_Years exists, normalize values like '4+' -> 4
if 'Stay_In_Current_City_Years' in df.columns:
    df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].astype(str).str.replace('\+', '')
    df['Stay_In_Current_City_Years'] = pd.to_numeric(df['Stay_In_Current_City_Years'], errors='coerce')
    df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].fillna(df['Stay_In_Current_City_Years'].mode()[0])

# City_Category, Gender, Age: fill with mode
for c in ['City_Category', 'Gender', 'Age']:
    if c in df.columns:
        df[c] = df[c].fillna(df[c].mode(dropna=True)[0])

# Any remaining numeric NaNs -> median
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for c in numeric_cols:
    if df[c].isna().sum() > 0:
        df[c] = df[c].fillna(df[c].median())

# -------------------------
# 5) ENCODING CATEGORICALS
# -------------------------
# We'll:
# - Keep 'Gender' as binary 0/1
# - One-hot encode City_Category
# - One-hot encode Age buckets
# - Keep Product_Category_x as numeric (they're often categorical numeric ids)

df_enc = df.copy()

# Binary encode Gender
if 'Gender' in df_enc.columns:
    df_enc['Gender'] = df_enc['Gender'].map({'F': 0, 'M': 1})
    # If some other values exist, fill with 0
    df_enc['Gender'] = df_enc['Gender'].fillna(0).astype(int)

# One-hot Age
if 'Age' in df_enc.columns:
    df_enc = pd.get_dummies(df_enc, columns=['Age'], prefix='Age')

# One-hot City_Category
if 'City_Category' in df_enc.columns:
    df_enc = pd.get_dummies(df_enc, columns=['City_Category'], prefix='City')

# One-hot Stay_In_Current_City_Years? Treat as numeric (already numeric after cleaning)

# Drop columns we will not use in X
non_feature_cols = ['User_ID', 'High_Value_Purchase', 'Purchase']
for c in non_feature_cols:
    if c not in df_enc.columns:
        # ignore if missing
        pass

# Final feature set
X = df_enc.drop(columns=[c for c in non_feature_cols if c in df_enc.columns])
y = df_enc['High_Value_Purchase']

print("Final feature matrix shape:", X.shape)

# -------------------------
# 6) TRAIN-TEST SPLIT
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y if y.nunique()>1 else None)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# -------------------------
# 7) SCALE NUMERICAL FEATURES
# -------------------------
# We'll scale all numeric features. If you have many one-hot columns, scaling them is harmless.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# 8) BASELINE: LOGISTIC REGRESSION
# -------------------------
log = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
log.fit(X_train_scaled, y_train)

pred_lr = log.predict(X_test_scaled)
pred_lr_proba = log.predict_proba(X_test_scaled)[:,1]

acc_lr = accuracy_score(y_test, pred_lr)
cm_lr = confusion_matrix(y_test, pred_lr)
print('\n--- Logistic Regression ---')
print('Accuracy:', acc_lr)
print('Confusion matrix:\n', cm_lr)
print('\nClassification report:\n', classification_report(y_test, pred_lr, digits=4))

# Feature importance (coefficients)
coefs = pd.Series(log.coef_[0], index=X.columns)
coefs_abs = coefs.abs().sort_values(ascending=False)
print('\nTop 10 features by absolute coefficient:')
print(coefs_abs.head(10))

# Plot top features
plt.figure(figsize=(8,6))
coefs_abs.head(10).sort_values().plot(kind='barh')
plt.title('Top 10 Logistic Regression feature importances (|coef|)')
plt.tight_layout()
plt.show()

# -------------------------
# 9) SIMPLE MLP (Keras)
# -------------------------
input_dim = X_train_scaled.shape[1]

mlp = keras.models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = mlp.fit(
    X_train_scaled, y_train,
    validation_split=0.1,
    epochs=15,
    batch_size=32,
    verbose=2
)

loss_mlp, acc_mlp = mlp.evaluate(X_test_scaled, y_test, verbose=0)
print('\n--- MLP (Keras) ---')
print(f'Test loss: {loss_mlp:.4f}, Test accuracy: {acc_mlp:.4f}')

# Plot training history (accuracy)
plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('MLP Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# Confusion matrix for MLP
pred_mlp = (mlp.predict(X_test_scaled)[:,0] >= 0.5).astype(int)
cm_mlp = confusion_matrix(y_test, pred_mlp)
print('MLP Confusion matrix:\n', cm_mlp)
print('\nClassification report (MLP):\n', classification_report(y_test, pred_mlp, digits=4))

# -------------------------
# 10) QUICK COMPARISON
# -------------------------
print('\nComparison:')
print(f'Logistic Regression Accuracy: {acc_lr:.4f}')
print(f'MLP Accuracy: {acc_mlp:.4f}')

# -------------------------
# 11) BONUS: permutation importance for logistic regression (to validate influential features)
# -------------------------
try:
    r = permutation_importance(log, X_test_scaled, y_test, n_repeats=20, random_state=RANDOM_STATE)
    perm_sorted_idx = r.importances_mean.argsort()[::-1]
    print('\nTop 10 features by permutation importance (logistic):')
    for i in perm_sorted_idx[:10]:
        print(f"{X.columns[i]}: mean_imp={r.importances_mean[i]:.4f} std={r.importances_std[i]:.4f}")
except Exception as e:
    print('Permutation importance failed:', e)

# -------------------------
# 12) SHORT INTERPRETATION (printed for convenience)
# -------------------------
print('\nInterpretation guidance:')
print('- Check which features have the largest absolute logistic coefficients and high permutation importance — these are strong predictors under the linear model.')
print('- If MLP outperforms LR, it may be capturing non-linear interactions between features (e.g., Age x Product_Category).')
print('- If LR performs similarly or better, dataset might be largely linearly separable or too small/noisy for the MLP to generalize.')
print('\nNext steps you can try:')
print('- Feature engineering: user-level aggregations (avg purchase, count of transactions, recency)')
print('- Class imbalance handling: up/down-sampling or class_weight in models')
print('- Hyperparameter tuning and deeper/wider networks or dropout for MLP')

# Save the trained models if desired
# mlp.save('mlp_model.h5')
# import joblib
# joblib.dump(log, 'logistic_model.joblib')

print('\nScript finished.')
