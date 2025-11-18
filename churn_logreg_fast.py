# final_churn_model.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)
import joblib
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------
# 1. Load Dataset
# ----------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Telco-Customer-Churn.csv"  # No local machine path

df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]

df["Churn"] = df["Churn"].str.strip().map({"Yes": 1, "No": 0})

# Remove customerID
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

# Clean TotalCharges
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    zero_mask = df["TotalCharges"].isna() & (df["tenure"] == 0)
    df.loc[zero_mask, "TotalCharges"] = 0
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# ----------------------------------------------
# 2. Preprocessing Setup
# ----------------------------------------------

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "Churn"]
categorical_cols = [c for c in df.columns if c not in numeric_cols and c != "Churn"]

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# ----------------------------------------------
# 3. Model Pipeline with GridSearchCV
# ----------------------------------------------

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, solver="liblinear"))
])

param_grid = {
    "clf__C": [0.01, 0.1, 1, 10]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# ----------------------------------------------
# 4. Evaluation
# ----------------------------------------------

y_proba = best_model.predict_proba(X_test)[:, 1]
prec, rec, thresh = precision_recall_curve(y_test, y_proba)
f_scores = 2 * (prec * rec) / (prec + rec + 1e-12)
best_idx = np.argmax(f_scores)
best_threshold = thresh[best_idx] if best_idx < len(thresh) else 0.5

y_pred = (y_proba >= best_threshold).astype(int)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba),
    "pr_auc": auc(rec, prec),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "best_threshold": float(best_threshold)
}

# ----------------------------------------------
# 5. Feature Importance
# ----------------------------------------------

preproc = best_model.named_steps["preprocessor"]
ohe = preproc.named_transformers_["cat"].named_steps["ohe"]
ohe_names = ohe.get_feature_names_out(categorical_cols)

feature_names = numeric_cols + ohe_names.tolist()
coefs = best_model.named_steps["clf"].coef_[0]

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coef": coefs,
    "abs_coef": np.abs(coefs)
}).sort_values("abs_coef", ascending=False)

# Save outputs
OUTPUT_DIR = BASE_DIR / "model_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

coef_df.to_csv(OUTPUT_DIR / "feature_coefficients.csv", index=False)
joblib.dump(best_model, OUTPUT_DIR / "final_model.pkl")

# ----------------------------------------------
# 6. Print Final Results
# ----------------------------------------------

print("\nModel Performance:")
for k, v in metrics.items():
    print(f"{k}: {v}")

print("\nTop 10 Features:")
print(coef_df.head(10).to_string(index=False))

print("\nFiles saved to:", OUTPUT_DIR)
