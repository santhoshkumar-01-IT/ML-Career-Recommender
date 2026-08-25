"""
Model Training & Evaluation Module
Trains and compares 5 Scikit-learn classification models:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors
- Support Vector Machine (SVC with Probability Calibration)

Evaluates true metrics (Accuracy, Precision, Recall, F1, Confusion Matrix),
selects the best model, and persists model artifacts using Joblib.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from preprocessing import CareerDataPreprocessor

def train_and_evaluate_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = os.path.join(base_dir, "data", "raw", "student_career_dataset.csv")
    
    if not os.path.exists(raw_data_path):
        from data_generator import generate_student_dataset
        print("Generating dataset...")
        df = generate_student_dataset(output_path=raw_data_path)
    else:
        df = pd.read_csv(raw_data_path)
        
    print(f"Loaded dataset with {len(df)} samples and {df.shape[1]} columns.")
    
    # 1. Feature / Target Separation
    X = df.drop(columns=["student_id", "career_label"])
    y = df["career_label"]
    
    # 2. Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples.")
    
    # 3. Fit Preprocessing Pipeline strictly on Training Data
    preprocessor = CareerDataPreprocessor()
    preprocessor.fit(X_train, y_train)
    
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    classes = sorted(list(df["career_label"].unique()))
    
    # Save processed splits
    processed_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    train_df_out = X_train.copy()
    train_df_out["career_label"] = y_train
    train_df_out.to_csv(os.path.join(processed_dir, "train_data.csv"), index=False)
    
    test_df_out = X_test.copy()
    test_df_out["career_label"] = y_test
    test_df_out.to_csv(os.path.join(processed_dir, "test_data.csv"), index=False)
    
    # 4. Define 5 ML Classifiers
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_split=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_split=4, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "Support Vector Machine": SVC(kernel="rbf", C=1.5, probability=True, random_state=42)
    }
    
    results = {}
    best_model_name = None
    best_f1_score = -1.0
    best_model_obj = None
    
    print("\n================ MODEL EVALUATION SUMMARY ================")
    
    for name, model in models.items():
        # Train
        model.fit(X_train_trans, y_train)
        
        # Predict on Test Set
        y_pred = model.predict(X_test_trans)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        p_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        p_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        r_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        r_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred, labels=classes).tolist()
        report_dict = classification_report(y_test, y_pred, target_names=classes, output_dict=True, zero_division=0)
        
        results[name] = {
            "accuracy": round(float(acc), 4),
            "precision_weighted": round(float(p_weighted), 4),
            "precision_macro": round(float(p_macro), 4),
            "recall_weighted": round(float(r_weighted), 4),
            "recall_macro": round(float(r_macro), 4),
            "f1_weighted": round(float(f1_weighted), 4),
            "f1_macro": round(float(f1_macro), 4),
            "confusion_matrix": cm,
            "classification_report": report_dict
        }
        
        print(f"[{name}] Accuracy: {acc*100:.2f}% | F1-Score (Weighted): {f1_weighted*100:.2f}% | F1-Score (Macro): {f1_macro*100:.2f}%")
        
        if f1_weighted > best_f1_score:
            best_f1_score = f1_weighted
            best_model_name = name
            best_model_obj = model
            
    print("===========================================================")
    print(f"\n🏆 Best Model Selected: {best_model_name} (Weighted F1: {best_f1_score*100:.2f}%)")
    
    # 5. Persist Best Model & Artifacts
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "career_model.pkl")
    preprocessor_path = os.path.join(models_dir, "preprocessing.pkl")
    metrics_path = os.path.join(models_dir, "model_metrics.json")
    
    joblib.dump(best_model_obj, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    
    metrics_payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "classes": classes,
        "best_model": best_model_name,
        "models": results
    }
    
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
        
    print(f"Saved best model to: {model_path}")
    print(f"Saved preprocessor to: {preprocessor_path}")
    print(f"Saved evaluation metrics to: {metrics_path}")
    
    return best_model_name, results

if __name__ == "__main__":
    train_and_evaluate_models()
