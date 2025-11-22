import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_auc_score
from tf_keras import models, layers
import warnings
warnings.filterwarnings('ignore')

# ============================================
# DATA PREPARATION
# ============================================

# Load your two datasets
df_purchase = pd.read_csv("C:/python/mock1/vivek_kumar_milestone 1/User_product_purchase_details_p2.csv")
df_user = pd.read_csv("C:/python/mock1/vivek_kumar_milestone 1/user_demographics.csv")

# Merge on User_ID
df = pd.merge(df_purchase, df_user, on="User_ID", how="left")

print("Dataset shape after merge:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Create binary target
df["High_Value_Purchase"] = (df["Purchase"] >= 10000).astype(int)

print("\nTarget distribution:")
print(df["High_Value_Purchase"].value_counts())

# Drop columns not needed
df = df.drop(["Product_ID", "User_ID"], axis=1)

# Handle missing values
print("\nMissing values before filling:")
print(df.isnull().sum())

df = df.fillna(0)

# Encode categorical variables using one-hot encoding
categorical_cols = ['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']

# Check which categorical columns exist in the dataframe
existing_categorical_cols = [col for col in categorical_cols if col in df.columns]

df = pd.get_dummies(df, columns=existing_categorical_cols, drop_first=True)

print("\nDataset shape after encoding:", df.shape)
print("\nColumn names after encoding:")
print(df.columns.tolist())

# Prepare features and target
X = df.drop(["High_Value_Purchase", "Purchase"], axis=1)
y = df["High_Value_Purchase"]

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# LOGISTIC REGRESSION BASELINE
# ============================================

print("\n" + "="*50)
print("LOGISTIC REGRESSION MODEL")
print("="*50)

log = LogisticRegression(max_iter=2000, random_state=42)
log.fit(X_train_scaled, y_train)

pred_lr = log.predict(X_test_scaled)
pred_lr_proba = log.predict_proba(X_test_scaled)[:, 1]

print("\nLR Accuracy:", accuracy_score(y_test, pred_lr))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred_lr))
print("\nClassification Report:")
print(classification_report(y_test, pred_lr))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': np.abs(log.coef_[0])
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# ============================================
# MLP (KERAS) MODEL
# ============================================

print("\n" + "="*50)
print("MLP NEURAL NETWORK MODEL")
print("="*50)

# Build MLP model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer="adam", 
    loss="binary_crossentropy", 
    metrics=["accuracy"]
)

# Display model architecture
print("\nModel Architecture:")
model.summary()

# Train model
print("\nTraining MLP...")
history = model.fit(
    X_train_scaled, 
    y_train, 
    epochs=20, 
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate on test set
loss, acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nMLP Test Accuracy: {acc:.4f}")
print(f"MLP Test Loss: {loss:.4f}")

# Get predictions for confusion matrix
pred_mlp = (model.predict(X_test_scaled) > 0.5).astype(int)
pred_mlp_proba = model.predict(X_test_scaled)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred_mlp))
print("\nClassification Report:")
print(classification_report(y_test, pred_mlp))

# ============================================
# MODEL COMPARISON
# ============================================

print("\n" + "="*50)
print("MODEL COMPARISON - ALL METRICS")
print("="*50)

# Calculate all metrics for both models
lr_accuracy = accuracy_score(y_test, pred_lr)
lr_precision = precision_score(y_test, pred_lr)
lr_recall = recall_score(y_test, pred_lr)
lr_f1 = f1_score(y_test, pred_lr)
lr_auc = roc_auc_score(y_test, pred_lr_proba)

mlp_accuracy = accuracy_score(y_test, pred_mlp)
mlp_precision = precision_score(y_test, pred_mlp)
mlp_recall = recall_score(y_test, pred_mlp)
mlp_f1 = f1_score(y_test, pred_mlp)
mlp_auc = roc_auc_score(y_test, pred_mlp_proba)

# Create comparison table
comparison_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Logistic Regression': [lr_accuracy, lr_precision, lr_recall, lr_f1, lr_auc],
    'MLP Neural Network': [mlp_accuracy, mlp_precision, mlp_recall, mlp_f1, mlp_auc],
    'Difference': [
        mlp_accuracy - lr_accuracy,
        mlp_precision - lr_precision,
        mlp_recall - lr_recall,
        mlp_f1 - lr_f1,
        mlp_auc - lr_auc
    ]
})

print("\n" + comparison_df.to_string(index=False))

print("\n" + "="*50)
print("SUMMARY")
print("="*50)

if mlp_accuracy > lr_accuracy:
    print("\n✓ MLP performed better overall!")
    print("Reason: Neural networks can capture non-linear relationships")
    print("between features that logistic regression cannot model.")
elif lr_accuracy > mlp_accuracy:
    print("\n✓ Logistic Regression performed better overall!")
    print("Reason: The relationship might be primarily linear, or the")
    print("neural network may be overfitting the training data.")
else:
    print("\n✓ Both models performed equally!")
    print("Reason: The problem might have simple linear patterns that")
    print("both models can capture effectively.")

print("\n" + "="*50)

print("End of analysis.")

















