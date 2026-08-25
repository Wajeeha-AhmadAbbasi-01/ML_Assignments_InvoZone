import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Create output directory for deliverables
output_dir = "benchmark_outputs"
os.makedirs(output_dir, exist_ok=True)


# Gets the directory where your current script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "processed_data", "procesed_customer_churn.csv")

# ==========================================
# 1. LOAD CLEAN CUSTOMER CHURN CSV
# ==========================================
print("=== Step 1: Loading Clean Dataset ===")


csv_path = "processed_data/procesed_customer_churn.csv" 
df = pd.read_csv(FILE_PATH)

print(f"Dataset Shape: {df.shape}")

# TODO: Change 'Churn' if your target column has a different name
target_col = 'Churn' 

# Classify features based on raw variable types
num_features = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).columns.tolist()
cat_features = df.drop(columns=[target_col]).select_dtypes(exclude=[np.number]).columns.tolist()

X = df.drop(columns=[target_col])
y = df[target_col]

# ==========================================
# 2. DATA SPLITTING & STRATIFICATION
# ==========================================
# Train (70%), Validation (15%), Test (15%) splits with stratification for imbalance
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1765, random_state=42, stratify=y_train_val
)

# ==========================================
# 3. PREPROCESSING PIPELINE
# ==========================================
num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
cat_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

# Processed splits for standard models
X_train_scaled = preprocessor.fit_transform(X_train)
X_val_scaled = preprocessor.transform(X_val)
X_test_scaled = preprocessor.transform(X_test)

# Unscaled pipeline variant specifically for the required SVM scaling experiment
preprocessor_unscaled = ColumnTransformer(transformers=[
    ('cat', cat_transformer, cat_features)
], remainder='passthrough') 

X_train_unscaled = preprocessor_unscaled.fit_transform(X_train)
X_val_unscaled = preprocessor_unscaled.transform(X_val)
X_test_unscaled = preprocessor_unscaled.transform(X_test)

# ==========================================
# 4. MODEL DICTIONARY SETUP (Required Benchmarks)
# ==========================================
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    
    # EXPERIMENT 1: Compare at least two values of K for KNN
    "KNN (K=3) [Low K]": KNeighborsClassifier(n_neighbors=3),
    "KNN (K=15) [High K]": KNeighborsClassifier(n_neighbors=15),
    
    # EXPERIMENT 2 & 5: Underfitting (Shallow) vs Overfitting (Deep) Decision Tree
    "Decision Tree (Shallow, Depth=3) [Underfit Model]": DecisionTreeClassifier(max_depth=3, random_state=42),
    "Decision Tree (Deep/Max, Depth=20) [Overfit Model]": DecisionTreeClassifier(max_depth=20, random_state=42),
    
    # EXPERIMENT 3: Compare Decision Tree with Random Forest
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    
    # EXPERIMENT 4: Compare Random Forest with a Boosting Model
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss'),
    
    # EXPERIMENT 6: Compare SVM with and without appropriate feature scaling
    "SVM (Scaled)": SVC(probability=True, random_state=42),
    "SVM (Unscaled)": SVC(probability=True, random_state=42)
}

# ==========================================
# 5. BENCHMARK EXECUTION ENGINE
# ==========================================
results_list = []

print("\n=== Step 2: Running Benchmarks ===")
for name, model in models.items():
    # Route to unscaled data if executing the specific unscaled SVM baseline
    X_tr = X_train_unscaled if "Unscaled" in name else X_train_scaled
    X_v = X_val_unscaled if "Unscaled" in name else X_val_scaled
    X_te = X_test_unscaled if "Unscaled" in name else X_test_scaled
    
    # Track Training Time
    start_train = time.time()
    model.fit(X_tr, y_train)
    train_time = time.time() - start_train
    
    # Track Inference Time 
    start_infer = time.time()
    val_preds = model.predict(X_v)
    infer_time = time.time() - start_infer
    
    # Extract probabilities
    val_probs = model.predict_proba(X_v)[:, 1] if hasattr(model, "predict_proba") else val_preds
    train_preds = model.predict(X_tr)
    test_preds = model.predict(X_te)
    
    metrics = {
        "Model / Experiment Configuration": name,
        "Train Acc": accuracy_score(y_train, train_preds),
        "Val Acc": accuracy_score(y_val, val_preds),
        "Val Precision": precision_score(y_val, val_preds, zero_division=0),
        "Val Recall": recall_score(y_val, val_preds),
        "Val F1": f1_score(y_val, val_preds),
        "Val ROC-AUC": roc_auc_score(y_val, val_probs),
        "Test Acc": accuracy_score(y_test, test_preds),
        "Train Time (s)": train_time,
        "Inference Time (s)": infer_time
    }
    results_list.append(metrics)

# ==========================================
# 6. EXPORT DELIVERABLES (CSV & PLOTS)
# ==========================================
results_df = pd.DataFrame(results_list)
csv_output_path = os.path.join(output_dir, "model_benchmark_results.csv")
results_df.to_csv(csv_output_path, index=False)
print(f"\n[SUCCESS] Saved clean results table to: {csv_output_path}")
print(results_df.to_string(index=False))

# Plot and save confusion matrices for direct homework submission requirements
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Subplot 1: Overfitting Deep Tree
cm_overfit = confusion_matrix(y_val, models["Decision Tree (Deep/Max, Depth=20) [Overfit Model]"].predict(X_val_scaled))
sns.heatmap(cm_overfit, annot=True, fmt='d', cmap='Oranges', ax=axes[0],
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
axes[0].set_title('Overfit Model: Deep Tree (Depth=20)')
axes[0].set_ylabel('Actual Label')
axes[0].set_xlabel('Predicted Label')

# Subplot 2: Best Performing Ensemble (XGBoost)
cm_best = confusion_matrix(y_val, models["XGBoost"].predict(X_val_scaled))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
axes[1].set_title('Top Predictive Model: XGBoost')
axes[1].set_ylabel('Actual Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout()
plot_output_path = os.path.join(output_dir, "experiment_confusion_matrices.png")
plt.savefig(plot_output_path, dpi=300)
plt.close()
print(f"[SUCCESS] Saved final evaluation plots to: {plot_output_path}\n")
