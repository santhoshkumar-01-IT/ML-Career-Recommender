"""
End-to-End Automated Test Suite
Tests data generation, preprocessing, model loading, top-3 predictions,
skill gap analysis, learning roadmap generation, and database operations.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add src to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from data_generator import generate_student_dataset, TECHNICAL_SKILLS, SOFT_SKILLS, INTERESTS, CAREERS
from preprocessing import CareerDataPreprocessor
from prediction import CareerPredictor
from skill_gap import SkillGapAnalyzer
from learning_recommendation import LearningRoadmapEngine
from database import db_manager

class TestCareerRecommendationSystem(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.predictor = CareerPredictor()
        cls.gap_analyzer = SkillGapAnalyzer()
        cls.roadmap_engine = LearningRoadmapEngine()
        
    def test_01_dataset_generation(self):
        df = generate_student_dataset(n_samples=160, random_state=42)
        self.assertEqual(len(df), 160)
        self.assertIn("career_label", df.columns)
        self.assertIn("cgpa", df.columns)
        for skill in TECHNICAL_SKILLS:
            self.assertIn(skill, df.columns)
            
    def test_02_preprocessor_validation(self):
        preprocessor = CareerDataPreprocessor()
        valid_input = {
            "name": "Test Student",
            "degree": "B.Sc Information Technology",
            "specialization": "Information Technology",
            "cgpa": 8.2,
            "interests": ["Data Analysis", "Machine Learning"],
            **{s: 4 for s in TECHNICAL_SKILLS},
            **{s: 4 for s in SOFT_SKILLS}
        }
        is_valid, msg = preprocessor.validate_single_input(valid_input)
        self.assertTrue(is_valid, f"Validation failed: {msg}")
        
        # Invalid CGPA
        invalid_input = valid_input.copy()
        invalid_input["cgpa"] = 12.5
        is_valid, msg = preprocessor.validate_single_input(invalid_input)
        self.assertFalse(is_valid)
        
    def test_03_prediction_top_3(self):
        # Sample Data Analyst profile
        student_data = {
            "name": "Ananya Sharma",
            "degree": "B.Sc Information Technology",
            "specialization": "Data Science",
            "cgpa": 8.5,
            "interests": ["Data Analysis", "Database Management"],
            "Python": 4, "Java": 1, "C_CPP": 0, "SQL": 5, "HTML_CSS": 1,
            "JavaScript": 1, "Excel": 5, "Power_BI": 5, "Statistics": 4,
            "Machine_Learning": 2, "Deep_Learning": 0, "Cloud_Computing": 2,
            "Networking": 1, "Cybersecurity": 1, "Git_GitHub": 3,
            "Communication": 4, "Problem_Solving": 4, "Teamwork": 4, "Leadership": 3, "Analytical_Thinking": 5
        }
        
        result = self.predictor.predict_top_careers(student_data, top_k=3)
        self.assertIn("primary_career", result)
        self.assertIn("top_recommendations", result)
        self.assertEqual(len(result["top_recommendations"]), 3)
        
        top1 = result["top_recommendations"][0]
        self.assertIn(top1["career"], CAREERS)
        self.assertGreaterEqual(top1["match_score"], 0.0)
        self.assertLessEqual(top1["match_score"], 100.0)
        print(f"\nTest Prediction Top 1: {top1['career']} with score {top1['match_score']}%")

    def test_04_skill_gap_analysis(self):
        student_skills = {
            "Python": 3, "SQL": 4, "Excel": 4, "Power_BI": 2, "Statistics": 3,
            "Machine_Learning": 1, "Deep_Learning": 0, "Java": 1, "C_CPP": 0,
            "HTML_CSS": 1, "JavaScript": 1, "Cloud_Computing": 1, "Networking": 1,
            "Cybersecurity": 1, "Git_GitHub": 2,
            "Communication": 4, "Problem_Solving": 4, "Teamwork": 3, "Leadership": 2, "Analytical_Thinking": 4
        }
        
        gap_report = self.gap_analyzer.analyze_gaps(student_skills, target_career="Data Analyst")
        self.assertEqual(gap_report["target_career"], "Data Analyst")
        self.assertGreaterEqual(gap_report["overall_readiness_score"], 0.0)
        self.assertLessEqual(gap_report["overall_readiness_score"], 100.0)
        self.assertIn("strong_skills", gap_report)
        self.assertIn("needs_improvement", gap_report)
        self.assertIn("major_gaps", gap_report)
        print(f"Skill Gap Readiness for Data Analyst: {gap_report['overall_readiness_score']}%")
        
    def test_05_learning_roadmap(self):
        student_skills = {s: 2 for s in TECHNICAL_SKILLS + SOFT_SKILLS}
        gap_report = self.gap_analyzer.analyze_gaps(student_skills, target_career="Data Scientist")
        roadmap = self.roadmap_engine.generate_roadmap(gap_report, target_career="Data Scientist")
        
        self.assertEqual(roadmap["target_career"], "Data Scientist")
        self.assertEqual(len(roadmap["phases"]), 4)
        self.assertGreater(roadmap["total_estimated_hours"], 0)
        print(f"Generated 4-Phase Learning Roadmap ({roadmap['total_estimated_hours']} hrs total)")

    def test_06_database_operations(self):
        student_info = {
            "student_id": "TEST_STU_001",
            "name": "Rahul Verma",
            "degree": "B.Sc Information Technology",
            "specialization": "Information Technology",
            "cgpa": 7.8,
            "academic_performance": "First Class"
        }
        skills = {s: 3 for s in TECHNICAL_SKILLS + SOFT_SKILLS}
        recs = [
            {"career": "Web Developer", "rank": 1, "match_score": 88.5},
            {"career": "Software Developer", "rank": 2, "match_score": 74.2},
            {"career": "Cloud Engineer", "rank": 3, "match_score": 52.0}
        ]
        
        saved_id = db_manager.save_assessment(student_info, skills, recs)
        self.assertEqual(saved_id, "TEST_STU_001")
        
        recent_df = db_manager.get_recent_assessments(limit=10)
        self.assertGreaterEqual(len(recent_df), 1)
        print(f"Database operation succeeded. Saved student ID: {saved_id}")

if __name__ == "__main__":
    unittest.main()
