"""
Data Preprocessing Pipeline Module
Handles input validation, feature engineering, multi-interest encoding, categorical one-hot encoding,
and numerical feature scaling without data leakage.
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

TECHNICAL_SKILLS = [
    "Python", "Java", "C_CPP", "SQL", "HTML_CSS", "JavaScript",
    "Excel", "Power_BI", "Statistics", "Machine_Learning",
    "Deep_Learning", "Cloud_Computing", "Networking", "Cybersecurity", "Git_GitHub"
]

SOFT_SKILLS = [
    "Communication", "Problem_Solving", "Teamwork", "Leadership", "Analytical_Thinking"
]

INTERESTS = [
    "Data Analysis", "Data Science", "Artificial Intelligence", "Machine Learning",
    "Web Development", "Software Development", "Database Management",
    "Cloud Computing", "Cybersecurity"
]

CAREERS = [
    "Data Analyst", "Data Scientist", "Software Developer", "Web Developer",
    "Database Administrator", "Cloud Engineer", "Cybersecurity Analyst", "Machine Learning Engineer"
]

def clean_column_name(name: str) -> str:
    return "interest_" + re.sub(r'[^a-zA-Z0-9]', '_', name.strip().lower())

class CareerDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Reusable, leakage-free Scikit-learn preprocessor pipeline for student career prediction.
    """
    def __init__(self):
        self.technical_skills = TECHNICAL_SKILLS
        self.soft_skills = SOFT_SKILLS
        self.all_interests = INTERESTS
        self.interest_columns = [clean_column_name(i) for i in INTERESTS]
        
        self.categorical_cols = ["degree", "specialization", "academic_performance"]
        self.numeric_cols = ["cgpa"] + TECHNICAL_SKILLS + SOFT_SKILLS
        
        self.scaler = StandardScaler()
        self.cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.label_encoder = LabelEncoder()
        
        self.feature_names_ = []
        self.is_fitted_ = False

    def validate_single_input(self, data: dict) -> tuple[bool, str]:
        """
        Validates individual student input from the Streamlit UI.
        """
        if not data.get("degree"):
            return False, "Degree is required."
        if not data.get("specialization"):
            return False, "Specialization is required."
            
        cgpa = data.get("cgpa")
        if cgpa is None:
            return False, "CGPA is required."
        try:
            cgpa_val = float(cgpa)
            if cgpa_val < 0.0 or cgpa_val > 10.0:
                return False, "CGPA must be between 0.0 and 10.0."
        except ValueError:
            return False, "CGPA must be a valid number."
            
        for skill in self.technical_skills + self.soft_skills:
            val = data.get(skill)
            if val is None:
                return False, f"Rating for '{skill}' is required."
            if not isinstance(val, (int, float)) or val < 0 or val > 5:
                return False, f"Skill rating for '{skill}' must be an integer between 0 and 5."
                
        interests = data.get("interests")
        if not interests or len(interests) == 0:
            return False, "Please select at least one career interest."
            
        return True, "Valid"

    def _extract_interest_matrix(self, interests_series: pd.Series) -> pd.DataFrame:
        """
        Encodes semi-colon / list separated interests into multi-hot binary indicator columns.
        """
        interest_matrix = []
        for val in interests_series:
            if isinstance(val, list):
                selected = set(val)
            elif isinstance(val, str):
                selected = set([s.strip() for s in val.split(";") if s.strip()])
            else:
                selected = set()
                
            row = [1 if interest in selected else 0 for interest in self.all_interests]
            interest_matrix.append(row)
            
        return pd.DataFrame(interest_matrix, columns=self.interest_columns)

    def fit(self, X: pd.DataFrame, y=None):
        X_df = X.copy()
        
        # 1. Multi-hot encode interests
        interest_df = self._extract_interest_matrix(X_df["interests"])
        
        # 2. Fit Categorical OneHotEncoder
        cat_df = X_df[self.categorical_cols].astype(str)
        self.cat_encoder.fit(cat_df)
        cat_feature_names = list(self.cat_encoder.get_feature_names_out(self.categorical_cols))
        
        # 3. Fit Numeric Scaler
        num_features = X_df[self.numeric_cols].values
        interest_features = interest_df.values
        combined_num_matrix = np.hstack([num_features, interest_features])
        
        self.scaler.fit(combined_num_matrix)
        
        self.feature_names_ = self.numeric_cols + self.interest_columns + cat_feature_names
        
        # 4. Fit LabelEncoder if target is present
        if y is not None:
            self.label_encoder.fit(y)
            
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise ValueError("CareerDataPreprocessor has not been fitted yet.")
            
        X_df = X.copy()
        
        # 1. Multi-hot encode interests
        interest_df = self._extract_interest_matrix(X_df["interests"])
        
        # 2. Transform Categorical
        cat_df = X_df[self.categorical_cols].astype(str)
        cat_encoded = self.cat_encoder.transform(cat_df)
        
        # 3. Transform Numeric & Interest flags
        num_features = X_df[self.numeric_cols].values
        interest_features = interest_df.values
        combined_num_matrix = np.hstack([num_features, interest_features])
        num_scaled = self.scaler.transform(combined_num_matrix)
        
        # 4. Combine all features
        X_transformed = np.hstack([num_scaled, cat_encoded])
        return X_transformed

    def transform_student_dict(self, student_dict: dict) -> np.ndarray:
        """
        Transforms a single student input dictionary into a 2D numpy array ready for model inference.
        """
        df = pd.DataFrame([student_dict])
        
        # Auto-compute academic performance if not provided
        if "academic_performance" not in df.columns or not df["academic_performance"].iloc[0]:
            cgpa = float(df["cgpa"].iloc[0])
            if cgpa >= 8.5:
                df["academic_performance"] = "Distinction"
            elif cgpa >= 7.0:
                df["academic_performance"] = "First Class"
            elif cgpa >= 6.0:
                df["academic_performance"] = "Second Class"
            else:
                df["academic_performance"] = "Pass"
                
        return self.transform(df)
