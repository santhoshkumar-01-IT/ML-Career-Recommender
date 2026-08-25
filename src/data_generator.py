"""
Dataset Generator Module
Generates realistic, reproducible synthetic student records for B.Sc. IT career classification.
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

INTERESTS = [
    "Data Analysis", "Data Science", "Artificial Intelligence", "Machine Learning",
    "Web Development", "Software Development", "Database Management",
    "Cloud Computing", "Cybersecurity"
]

CAREERS = [
    "Data Analyst", "Data Scientist", "Software Developer", "Web Developer",
    "Database Administrator", "Cloud Engineer", "Cybersecurity Analyst", "Machine Learning Engineer"
]

def get_academic_performance(cgpa: float) -> str:
    if cgpa >= 8.5:
        return "Distinction"
    elif cgpa >= 7.0:
        return "First Class"
    elif cgpa >= 6.0:
        return "Second Class"
    else:
        return "Pass"

def generate_student_dataset(n_samples: int = 1600, random_state: int = 42, output_path: str = None) -> pd.DataFrame:
    """
    Generates a realistic synthetic dataset for student career recommendation.
    
    Academic Disclaimer:
    This dataset contains synthetic/demo student records generated with a fixed random seed (seed=42)
    for academic modeling, training, and benchmarking purposes.
    """
    np.random.seed(random_state)
    
    current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.path.join(base_dir, "src")
    config_path = os.path.join(os.path.dirname(current_dir), "config", "career_requirements.json")
    
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            career_benchmarks = json.load(f).get("careers", {})
    else:
        career_benchmarks = {}

    records = []
    samples_per_career = n_samples // len(CAREERS)
    
    degrees = ["B.Sc Information Technology", "B.Sc Computer Science", "BCA"]
    degree_probs = [0.80, 0.15, 0.05]
    specializations = ["Information Technology", "Data Science", "Software Engineering", "Cloud & Security", "Web Technologies"]
    
    for career_idx, career in enumerate(CAREERS):
        benchmark = career_benchmarks.get(career, {})
        req_tech = benchmark.get("technical_skills", {})
        req_soft = benchmark.get("soft_skills", {s: 3 for s in SOFT_SKILLS})
        primary_interests = benchmark.get("primary_interests", [])
        
        for i in range(samples_per_career):
            student_id = f"STU{1000 + career_idx * samples_per_career + i + 1}"
            degree = np.random.choice(degrees, p=degree_probs)
            specialization = np.random.choice(specializations)
            
            # CGPA calculation with realistic distribution
            base_cgpa = 7.6 if career in ["Data Scientist", "Machine Learning Engineer"] else 7.2
            cgpa = np.clip(np.random.normal(base_cgpa, 0.85), 5.5, 9.9)
            cgpa = round(float(cgpa), 2)
            academic_perf = get_academic_performance(cgpa)
            
            # Technical Skills (0-5)
            student_tech = {}
            for skill in TECHNICAL_SKILLS:
                target_level = req_tech.get(skill, 1)
                noise = np.random.choice([-2, -1, 0, 1], p=[0.08, 0.22, 0.55, 0.15])
                val = int(np.clip(target_level + noise, 0, 5))
                student_tech[skill] = val
                
            # Soft Skills (0-5)
            student_soft = {}
            for skill in SOFT_SKILLS:
                target_level = req_soft.get(skill, 3)
                noise = np.random.choice([-1, 0, 1], p=[0.20, 0.60, 0.20])
                val = int(np.clip(target_level + noise, 0, 5))
                student_soft[skill] = val
                
            # Interests (multi-select)
            selected_interests = list(primary_interests)
            other_interests = [inte for inte in INTERESTS if inte not in primary_interests]
            extra_count = np.random.choice([0, 1, 2], p=[0.25, 0.50, 0.25])
            if extra_count > 0 and len(other_interests) >= extra_count:
                extra_chosen = np.random.choice(other_interests, size=extra_count, replace=False).tolist()
                selected_interests.extend(extra_chosen)
            
            interests_str = "; ".join(selected_interests)
            
            record = {
                "student_id": student_id,
                "degree": degree,
                "specialization": specialization,
                "cgpa": cgpa,
                "academic_performance": academic_perf,
                **student_tech,
                **student_soft,
                "interests": interests_str,
                "career_label": career
            }
            records.append(record)
            
    df = pd.DataFrame(records)
    # Shuffle
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    if output_path is None:
        target_dir = os.path.dirname(current_dir)
        output_path = os.path.join(target_dir, "data", "raw", "student_career_dataset.csv")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated: {len(df)} records saved to {output_path}")
    return df

if __name__ == "__main__":
    generate_student_dataset()
