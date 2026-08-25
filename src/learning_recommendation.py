"""
Learning Recommendation Engine Module
Transforms identified skill gaps into an actionable, 4-phase personalized learning roadmap
with curated free courses, industry certifications, practical portfolio projects, and estimated study timelines.
"""

import os
import json

class LearningRoadmapEngine:
    """
    Generates tailored learning pathways based on a student's evaluated skill gaps.
    """
    def __init__(self, resources_path: str = None):
        if resources_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            resources_path = os.path.join(base_dir, "config", "learning_resources.json")
            
        self.resources_path = resources_path
        self.resources_data = {}
        self._load_resources()
        
    def _load_resources(self):
        if os.path.exists(self.resources_path):
            with open(self.resources_path, "r", encoding="utf-8") as f:
                self.resources_data = json.load(f).get("skills", {})
        else:
            raise FileNotFoundError(f"Learning resources config not found at: {self.resources_path}")

    def generate_roadmap(self, gap_report: dict, target_career: str) -> dict:
        """
        Builds a 4-phase structured curriculum for the student to bridge skill gaps for target_career.
        """
        major_gaps = gap_report.get("major_gaps", [])
        needs_improvement = gap_report.get("needs_improvement", [])
        strong_skills = gap_report.get("strong_skills", [])
        
        # Skill item processing
        def get_skill_plan(skill_item, priority: str):
            skill_name = skill_item["skill"]
            student_lvl = skill_item["student_level"]
            req_lvl = skill_item["required_level"]
            res = self.resources_data.get(skill_name, {})
            
            topics_to_learn = []
            if student_lvl <= 1:
                topics_to_learn.extend(res.get("beginner", []))
            if student_lvl <= 3 and req_lvl >= 3:
                topics_to_learn.extend(res.get("intermediate", []))
            if student_lvl <= 4 and req_lvl >= 4:
                topics_to_learn.extend(res.get("advanced", []))
                
            if not topics_to_learn:
                topics_to_learn = res.get("intermediate", []) + res.get("advanced", [])
                
            hours = res.get("estimated_hours", 20)
            if priority == "High":
                hours = int(hours * 1.2)
            else:
                hours = int(hours * 0.7)
                
            return {
                "skill": skill_name,
                "category": skill_item.get("category_type", "Technical"),
                "priority": priority,
                "current_level": student_lvl,
                "target_level": req_lvl,
                "topics": topics_to_learn[:4],
                "courses": res.get("courses", []),
                "projects": res.get("projects", []),
                "estimated_hours": max(hours, 10)
            }

        high_priority_items = [get_skill_plan(item, "High") for item in major_gaps]
        medium_priority_items = [get_skill_plan(item, "Medium") for item in needs_improvement]
        
        total_hours = sum(p["estimated_hours"] for p in high_priority_items + medium_priority_items)
        
        # 4-Phase Chronological Timeline
        phase_1_items = [p for p in high_priority_items if p["category"] == "Technical"]
        phase_2_items = [p for p in medium_priority_items if p["category"] == "Technical"]
        phase_4_soft = [p for p in high_priority_items + medium_priority_items if p["category"] == "Soft"]
        
        # Phase 3 Project Selection based on target career
        career_projects = {
            "Data Analyst": "End-to-End Sales Analytics & Customer Churn Dashboard with SQL, Python, and Power BI",
            "Data Scientist": "Predictive Health Risk Classification & Explanatory Analytics using Scikit-Learn and SHAP",
            "Software Developer": "Scalable RESTful Microservices Backend with Java Spring Boot / Python FastAPI and MySQL",
            "Web Developer": "Full-Stack Collaborative Project Management Web Application with React and Node/Python",
            "Database Administrator": "High-Availability MySQL Database Cluster with Automated Backup, Replication, and Tuning",
            "Cloud Engineer": "Serverless Multi-Tier Cloud Deployment with Terraform, AWS Lambda / GCP Cloud Run, and CI/CD",
            "Cybersecurity Analyst": "Automated Network Vulnerability Scanner, Log Audit Pipeline, and SIEM Alert System",
            "Machine Learning Engineer": "Containerized ML Model Serving Pipeline with FastAPI, Docker, and Automated Monitoring"
        }
        
        phases = [
            {
                "phase_number": 1,
                "title": "Phase 1: Critical Foundations & Major Gaps",
                "duration": "Weeks 1 ? 4 (12?15 hrs/week)",
                "description": "Close critical blockers and master fundamental concepts required for advanced coursework.",
                "skills": phase_1_items if phase_1_items else [{"skill": "Prerequisites Review", "topics": ["Review foundational IT mathematics & coding standards"], "estimated_hours": 15}]
            },
            {
                "phase_number": 2,
                "title": "Phase 2: Core Domain Competencies",
                "duration": "Weeks 5 ? 8 (10?12 hrs/week)",
                "description": "Elevate intermediate skills to meet industry production benchmarks.",
                "skills": phase_2_items if phase_2_items else [{"skill": "Advanced Domain Topics", "topics": ["Production-level frameworks and best practices"], "estimated_hours": 15}]
            },
            {
                "phase_number": 3,
                "title": "Phase 3: Portfolio Project & Applied Engineering",
                "duration": "Weeks 9 ? 12 (10?15 hrs/week)",
                "description": "Build and deploy a capstone portfolio project demonstrating complete domain mastery.",
                "capstone_project": career_projects.get(target_career, "Industry-standard Capstone Project for " + target_career),
                "skills": [{"skill": "Portfolio Engineering", "topics": ["Git Version Control", "Documentation & Architecture Diagrams", "Deployment & Live Demo"], "estimated_hours": 30}]
            },
            {
                "phase_number": 4,
                "title": "Phase 4: Soft Skills, System Architecture & Viva/Interview Prep",
                "duration": "Weeks 13 ? 14 (6?8 hrs/week)",
                "description": "Refine technical communication, problem solving, system design, and final presentation readiness.",
                "skills": phase_4_soft if phase_4_soft else [{"skill": "Technical Interview & Viva Prep", "topics": ["Mock Technical Interviews", "Project Walkthrough Presentation", "Problem Solving Practice"], "estimated_hours": 15}]
            }
        ]
        
        return {
            "target_career": target_career,
            "total_estimated_hours": total_hours if total_hours > 0 else 45,
            "recommended_weekly_hours": 12,
            "total_weeks": 14,
            "high_priority_count": len(high_priority_items),
            "medium_priority_count": len(medium_priority_items),
            "high_priority_skills": high_priority_items,
            "medium_priority_skills": medium_priority_items,
            "phases": phases,
            "capstone_project": career_projects.get(target_career, "Capstone Portfolio Project")
        }
