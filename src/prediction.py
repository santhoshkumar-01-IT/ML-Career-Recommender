"""
Career Prediction Engine Module
Loads persisted ML model and preprocessor pipeline to produce ranked top-3 career recommendations
with calibrated real-world confidence scores and probabilistic match percentages.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

class CareerPredictor:
    """
    Inference engine for IT career recommendations.
    """
    def __init__(self, model_path: str = None, preprocessor_path: str = None, config_path: str = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if model_path is None:
            model_path = os.path.join(base_dir, "models", "career_model.pkl")
        if preprocessor_path is None:
            preprocessor_path = os.path.join(base_dir, "models", "preprocessing.pkl")
        if config_path is None:
            config_path = os.path.join(base_dir, "config", "career_requirements.json")
            
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.config_path = config_path
        
        self.model = None
        self.preprocessor = None
        self.career_info = {}
        
        self._load_artifacts()
        
    def _load_artifacts(self):
        if os.path.exists(self.model_path) and os.path.exists(self.preprocessor_path):
            self.model = joblib.load(self.model_path)
            self.preprocessor = joblib.load(self.preprocessor_path)
        else:
            raise FileNotFoundError("Trained model or preprocessor artifact not found. Please run train_model.py first.")
            
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.career_info = json.load(f).get("careers", {})

    def predict_top_careers(self, student_data: dict, top_k: int = 3) -> dict:
        """
        Predicts top K careers with calculated model probabilities.
        """
        if self.model is None or self.preprocessor is None:
            self._load_artifacts()
            
        # 1. Validate Input
        is_valid, err_msg = self.preprocessor.validate_single_input(student_data)
        if not is_valid:
            raise ValueError(f"Input validation error: {err_msg}")
            
        # 2. Transform Features
        X_trans = self.preprocessor.transform_student_dict(student_data)
        
        # 3. Model Inference (Probabilities)
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(X_trans)[0]
            classes = self.model.classes_
        else:
            # Fallback for models without direct predict_proba: softmax over decision function
            decision = self.model.decision_function(X_trans)[0]
            exp_d = np.exp(decision - np.max(decision))
            probas = exp_d / np.sum(exp_d)
            classes = self.model.classes_
            
        # Build full score mapping
        all_career_scores = {
            classes[i]: round(float(probas[i]) * 100, 2)
            for i in range(len(classes))
        }
        
        # Rank descending
        sorted_indices = np.argsort(probas)[::-1]
        
        top_recommendations = []
        for rank_idx, idx in enumerate(sorted_indices[:top_k]):
            career_name = classes[idx]
            prob = float(probas[idx])
            score_pct = round(prob * 100, 2)
            desc = self.career_info.get(career_name, {}).get("description", "IT Career Path")
            
            top_recommendations.append({
                "rank": rank_idx + 1,
                "career": career_name,
                "match_score": score_pct,
                "probability": round(prob, 4),
                "description": desc,
                "is_primary": (rank_idx == 0)
            })
            
        return {
            "primary_career": top_recommendations[0]["career"],
            "top_recommendations": top_recommendations,
            "all_scores": all_career_scores,
            "model_type": type(self.model).__name__
        }
