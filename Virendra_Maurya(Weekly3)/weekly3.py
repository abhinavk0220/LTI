# ============================================
# Project: High Value Purchase Prediction
# Model: Logistic Regression + MLP Neural Network
# Author: Virendra Maurya
# ============================================

# -------- Step 1: Import Libraries ----------
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras import models, layers


# -------- Step 2: Load the datasets ---------
df_purchase = pd.read_csv("User_product_purchase_details_p2.csv")
df_user = pd.read_csv("user_demographics.csv")

print("Purchase Dataset Shape:", df_purchase.shape)
print("User Dataset Shape:", df_user.shape)


# -------- Step 3: Merge datasets on User_ID --------\
df = pd.merge(df_purchase, df_user, on="User_ID", how="left")
print("Merged Dataset Shape:", df.shape)


# -------- Step 4: Create binary target variable --------
# Target = 1 if Purchase >= 10000 else 0
df["High_Value_Purchase"] = (df["Purchase"] >= 10000).astype(int)


# -------- Step 5: Drop unnecessary columns --------
# Product_ID is not useful for prediction
df = df.drop(["Product_ID"], axis=1)


# -------- Step 6: Handle missing values --------
df = df.fillna(0)


# -------- Step 7: Encode categorical variables --------
# One-hot encoding (convert categories into 0/1 columns)
df = pd.get_dummies(df, drop_first=True)


# -------- Step 8: Split features (X) and target (y) --------
X = df.drop(["High_Value_Purchase", "Purchase"], axis=1)
y = df["High_Value_Purchase"]


# -------- Step 9: Train-Test Split --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# -------- Step 10: Scale the numerical features --------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =====================================================
#                   MODEL 1: LOGISTIC REGRESSION
# =====================================================

print("\n================ LOGISTIC REGRESSION ================\n")

log_model = LogisticRegression(max_iter=2000)

# Train model
log_model.fit(X_train_scaled, y_train)

# Predict
pred_lr = log_model.predict(X_test_scaled)

# Evaluate
lr_acc = accuracy_score(y_test, pred_lr)
lr_cm = confusion_matrix(y_test, pred_lr)

print("Logistic Regression Accuracy:", lr_acc)
print("Confusion Matrix:\n", lr_cm)
print("Classification Report:\n", classification_report(y_test, pred_lr))


# =====================================================
#                   MODEL 2: MLP NEURAL NETWORK
# =====================================================

print("\n================ MLP NEURAL NETWORK ================\n")

model = models.Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train model
history = model.fit(X_train_scaled, y_train, epochs=15, batch_size=32, verbose=1)

# Evaluate
loss, mlp_acc = model.evaluate(X_test_scaled, y_test)
print("MLP Test Accuracy:", mlp_acc)


# =====================================================
#                   FINAL COMPARISON
# =====================================================

print("\n================ FINAL COMPARISON ================\n")
print("Logistic Regression Accuracy:", lr_acc)
print("MLP Neural Network Accuracy:", mlp_acc)

if mlp_acc > lr_acc:
    print("➡ MLP performed better!")
else:
    print("➡ Logistic Regression performed better!")
print(classification_report(y_test, (model.predict(X_test_scaled) > 0.5).astype(int)))

