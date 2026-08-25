"""
Skill Gap Analysis Engine Module
Compares student technical and soft skill proficiencies against structured industry benchmarks,
categorizes proficiencies (Strong, Meets Requirement, Needs Improvement, Major Gap),
and calculates multidimensional Career Readiness Scores.
"""

import os
import json
import numpy as np
import pandas as pd

TECHNICAL_SKILLS = [
    "Python", "Java", "C_CPP", "SQL", "HTML_CSS", "JavaScript",
    "Excel", "Power_BI", "Statistics", "Machine_Learning",
    "Deep_Learning", "Cloud_Computing", "Networking", "Cybersecurity", "Git_GitHub"
]

SOFT_SKILLS = [
    "Communication", "Problem_Solving", "Teamwork", "Leadership", "Analytical_Thinking"
]

class SkillGapAnalyzer:
    """
    Performs comprehensive gap analysis between a student's profile and target career requirements.
    """
    def __init__(self, config_path: str = None):
        if config_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, "config", "career_requirements.json")
            
        self.config_path = config_path
        self.career_requirements = {}
        self._load_benchmarks()
        
    def _load_benchmarks(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.career_requirements = json.load(f).get("careers", {})
        else:
            raise FileNotFoundError(f"Career requirements config not found at: {self.config_path}")

    def analyze_gaps(self, student_skills: dict, target_career: str) -> dict:
        """
        Analyzes technical and soft skill gaps for a specified career path.
        """
        if target_career not in self.career_requirements:
            raise ValueError(f"Career '{target_career}' is not defined in benchmark configurations.")
            
        benchmark = self.career_requirements[target_career]
        req_tech = benchmark.get("technical_skills", {})
        req_soft = benchmark.get("soft_skills", {})
        
        all_skills_analysis = []
        strong_skills = []
        meets_skills = []
        needs_improvement = []
        major_gaps = []
        
        total_req_tech_points = 0
        total_achieved_tech_points = 0
        total_req_soft_points = 0
        total_achieved_soft_points = 0
        
        # 1. Technical Skills Evaluation
        for skill in TECHNICAL_SKILLS:
            req_lvl = int(req_tech.get(skill, 0))
            stud_lvl = int(student_skills.get(skill, 0))
            gap = stud_lvl - req_lvl
            
            # Points for readiness
            total_req_tech_points += req_lvl
            total_achieved_tech_points += min(stud_lvl, req_lvl)
            
            # Classification
            if req_lvl == 0 and stud_lvl > 0:
                category = "Strong"
            elif (stud_lvl >= req_lvl + 1 and req_lvl > 0) or (stud_lvl == 5 and req_lvl >= 3):
                category = "Strong"
            elif stud_lvl == req_lvl:
                category = "Meets Requirement"
            elif (req_lvl >= 4 and stud_lvl <= 1) or (gap <= -3):
                category = "Major Gap"
            elif gap in [-1, -2] and req_lvl > 0:
                category = "Needs Improvement"
            else:
                category = "Meets Requirement"
                
            item = {
                "skill": skill,
                "category_type": "Technical",
                "student_level": stud_lvl,
                "required_level": req_lvl,
                "gap": gap,
                "status": category
            }
            
            all_skills_analysis.append(item)
            if category == "Strong":
                strong_skills.append(item)
            elif category == "Meets Requirement":
                meets_skills.append(item)
            elif category == "Needs Improvement":
                needs_improvement.append(item)
            elif category == "Major Gap":
                major_gaps.append(item)
                
        # 2. Soft Skills Evaluation
        for skill in SOFT_SKILLS:
            req_lvl = int(req_soft.get(skill, 3))
            stud_lvl = int(student_skills.get(skill, 3))
            gap = stud_lvl - req_lvl
            
            total_req_soft_points += req_lvl
            total_achieved_soft_points += min(stud_lvl, req_lvl)
            
            if stud_lvl >= req_lvl + 1 or (stud_lvl == 5 and req_lvl >= 3):
                category = "Strong"
            elif stud_lvl == req_lvl:
                category = "Meets Requirement"
            elif gap <= -2:
                category = "Major Gap"
            else:
                category = "Needs Improvement"
                
            item = {
                "skill": skill,
                "category_type": "Soft",
                "student_level": stud_lvl,
                "required_level": req_lvl,
                "gap": gap,
                "status": category
            }
            
            all_skills_analysis.append(item)
            if category == "Strong":
                strong_skills.append(item)
            elif category == "Meets Requirement":
                meets_skills.append(item)
            elif category == "Needs Improvement":
                needs_improvement.append(item)
            elif category == "Major Gap":
                major_gaps.append(item)

        # 3. Calculate Readiness Percentages
        tech_readiness = (total_achieved_tech_points / total_req_tech_points * 100) if total_req_tech_points > 0 else 100.0
        soft_readiness = (total_achieved_soft_points / total_req_soft_points * 100) if total_req_soft_points > 0 else 100.0
        overall_readiness = 0.70 * tech_readiness + 0.30 * soft_readiness
        
        # 4. Prepare Plotly Chart Data
        # Filter for relevant skills (where required level > 0 or student level > 0)
        relevant_tech = [s for s in all_skills_analysis if s["category_type"] == "Technical" and (s["required_level"] > 0 or s["student_level"] > 0)]
        
        radar_labels = [s["skill"] for s in relevant_tech]
        radar_student = [s["student_level"] for s in relevant_tech]
        radar_required = [s["required_level"] for s in relevant_tech]
        
        bar_skills = [s["skill"] for s in all_skills_analysis if s["category_type"] == "Technical" and s["required_level"] > 0]
        bar_gaps = [s["gap"] for s in all_skills_analysis if s["category_type"] == "Technical" and s["required_level"] > 0]
        
        return {
            "target_career": target_career,
            "career_description": benchmark.get("description", ""),
            "overall_readiness_score": round(float(overall_readiness), 1),
            "technical_readiness": round(float(tech_readiness), 1),
            "soft_readiness": round(float(soft_readiness), 1),
            "counts": {
                "strong": len(strong_skills),
                "meets": len(meets_skills),
                "needs_improvement": len(needs_improvement),
                "major_gap": len(major_gaps)
            },
            "strong_skills": strong_skills,
            "meets_skills": meets_skills,
            "needs_improvement": needs_improvement,
            "major_gaps": major_gaps,
            "all_skills": all_skills_analysis,
            "radar_chart": {
                "labels": radar_labels,
                "student_levels": radar_student,
                "required_levels": radar_required
            },
            "bar_chart": {
                "skills": bar_skills,
                "gaps": bar_gaps
            }
        }
