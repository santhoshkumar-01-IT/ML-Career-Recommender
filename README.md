# Machine Learning-Based Career Recommendation and Skill Gap Analysis System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ml-career-recommender.streamlit.app)
[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![Database](https://img.shields.io/badge/Database-MySQL%20%7C%20SQLite-00758F.svg)](https://www.mysql.com/)
[![License](https://img.shields.io/badge/Academic-Project-green.svg)](#)

A comprehensive, end-to-end Machine Learning web application engineered for **students, graduates, and tech career seekers**. The system evaluates educational qualifications, 15 technical skills, 5 soft skills, and 9 domain interests to recommend top IT career pathways, compute probabilistic match percentages, evaluate skill gaps against industry benchmarks, and generate personalized 4-phase learning roadmaps.

> 🚀 **Live Demo Web Link:** [https://ml-career-recommender.streamlit.app](https://ml-career-recommender.streamlit.app)

---

## Table of Contents
1. [Project Overview](#-project-overview)
2. [Key Objectives & Features](#-key-objectives--features)
3. [Technology Stack](#-technology-stack)
4. [System Architecture & Workflow](#-system-architecture--workflow)
5. [Dataset Description & Academic Disclaimer](#-dataset-description--academic-disclaimer)
6. [Data Preprocessing Pipeline](#-data-preprocessing-pipeline)
7. [Machine Learning Methodology & Evaluation](#-machine-learning-methodology--evaluation)
8. [Skill Gap Engine & Mathematical Formulations](#-skill-gap-engine--mathematical-formulations)
9. [Personalized Learning Roadmap Engine](#-personalized-learning-roadmap-engine)
10. [Database Design (MySQL & Dual-Mode SQLite)](#-database-design)
11. [Project Structure](#-project-structure)
12. [Installation & Setup Instructions](#-installation--setup-instructions)
13. [Step-by-Step Execution Guide](#-step-by-step-execution-guide)
14. [Viva & Project Review Guide (Academic Defense Q&A)](#-viva--project-review-guide)
15. [Academic Disclaimer](#-academic-disclaimer)

---

##  Project Overview

Transitioning from academic coursework in Information Technology to specialized industry roles is a significant hurdle for students due to:
- Lack of clarity regarding prerequisite skills for distinct career tracks.
- Ambiguity in identifying specific skill deficiencies.
- Disconnected online learning pathways lacking priority and structure.

This system solves these challenges by combining **multi-class supervised Machine Learning classification**, **dynamic benchmark comparison**, and **automated learning roadmap synthesis** inside a clean, modern Streamlit dashboard.

---

##  Key Objectives & Features

1. **Student Assessment Module**:
   - Captures Degree, Specialization, CGPA (0-10.0), and Academic Standing.
   - Collects 0-5 integer ratings across 15 Technical Skills and 5 Soft Skills.
   - Captures multi-select domain interests.
   - Real-time client-side validation and quick-demo preset profiles.

2. **Ranked Multi-Class Career Prediction**:
   - Predicts and ranks the **Top 3 suitable IT careers** with true probability percentages.
   - Evaluates across 8 core career paths:
     1. Data Analyst
     2. Data Scientist
     3. Software Developer
     4. Web Developer
     5. Database Administrator
     6. Cloud Engineer
     7. Cybersecurity Analyst
     8. Machine Learning Engineer

3. **Configurable Skill Gap Engine**:
   - Compares student ratings against JSON-configured industry benchmark requirements.
   - Classifies competencies into: `Strong`, `Meets Requirement`, `Needs Improvement`, and `Major Gap`.
   - Computes weighted **Career Readiness Index (%)**, Technical Readiness, and Soft Skills Readiness.
   - Visualizes discrepancies using interactive Plotly Radar and Gap Bar charts.

4. **Personalized 4-Phase Learning Path**:
   - Generates an actionable chronological curriculum:
     - **Phase 1: Critical Foundations & Major Gaps (Weeks 1?4)**
     - **Phase 2: Core Domain Competencies (Weeks 5?8)**
     - **Phase 3: Capstone Portfolio Project (Weeks 9?12)**
     - **Phase 4: Soft Skills & Technical Viva/Interview Prep (Weeks 13?14)**
   - Maps each skill to curated free courses (Coursera, freeCodeCamp, Kaggle, Microsoft Learn), topics, and hands-on projects.

5. **Multi-Model ML Performance Visualizer**:
   - Real benchmark comparison of 5 classifiers: Logistic Regression, Decision Tree, Random Forest, KNN, and SVM.
   - Displays true test accuracy, precision, recall, F1-scores, and interactive Confusion Matrix heatmaps.

6. **Dual-Mode Database Persistence**:
   - Standard MySQL schema with foreign keys and index constraints.
   - Automatic fallback to local SQLite (`data/career_recommendations.db`) when external MySQL servers are unconfigured or offline.
   - Full assessment history audit and CSV export capability.

---

##  Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend / Web UI** | Streamlit 1.35+, Custom CSS |
| **Programming Language** | Python 3.11+ |
| **Machine Learning** | Scikit-learn 1.4+, Joblib |
| **Data Processing & Math** | Pandas, NumPy |
| **Data Visualization** | Plotly Express & Graph Objects |
| **Database** | MySQL 8.0+ (PyMySQL) with automatic SQLite local fallback |
| **Environment Management** | Python `venv`, `python-dotenv` |

---

## ? System Architecture & Workflow

```
[Student Profile Input] (Degree, CGPA, 15 Tech Skills, 5 Soft Skills, Interests)
            ?
            ?
[Data Validation & Preprocessing Pipeline] (Range checks, One-Hot Encoding, Multi-Hot Interests, Standard Scaling)
            ?
            ?
[Machine Learning Classification Engine] (Random Forest / Logistic Regression / SVM)
            ?
            
            ?                                                  ?
[Top 3 Ranked Recommendations]                    [Skill Gap Analyzer Engine]
 (Calibrated Probabilities & Match %)              (Benchmark Matrix Comparison)
            ?                                                  ?
            ?                                                  ?
            ?                                     [Career Readiness Index %]
            ?                                     (Strong, Meets, Needs Improvement, Major Gap)
            ?                                                  ?
            
                                      ?
                        [Personalized Learning Path Engine]
                         (4-Phase Action Plan & Capstone Project)
                                      ?
                                      ?
                        [Dual-Mode Database & Streamlit UI]
                         (Audit Logging to MySQL/SQLite + CSV Export)
```

---

##  Dataset Description & Academic Disclaimer

> **Academic Project Disclaimer:**
> The training dataset contains **1,600 synthetic student records** generated with a fixed random seed (`random_state=42`) using `src/data_generator.py`. It is designed strictly for academic demonstration, algorithmic benchmarking, and model evaluation.

### Feature Schema:
- `student_id`: Unique identifier (e.g., `STU1001`)
- `degree`: Academic degree (`B.Sc Information Technology`, `B.Sc Computer Science`, `BCA`)
- `specialization`: Academic focus (`Information Technology`, `Data Science`, `Software Engineering`, etc.)
- `cgpa`: Cumulative Grade Point Average (5.50 ? 10.00)
- `academic_performance`: Categorical standing (`Distinction`, `First Class`, `Second Class`, `Pass`)
- **15 Technical Skills (0?5)**: `Python`, `Java`, `C_CPP`, `SQL`, `HTML_CSS`, `JavaScript`, `Excel`, `Power_BI`, `Statistics`, `Machine_Learning`, `Deep_Learning`, `Cloud_Computing`, `Networking`, `Cybersecurity`, `Git_GitHub`
- **5 Soft Skills (0?5)**: `Communication`, `Problem_Solving`, `Teamwork`, `Leadership`, `Analytical_Thinking`
- `interests`: Semicolon-separated list of selected career interests (9 domains)
- `career_label` *(Target)*: One of the 8 supported career categories (balanced 200 records per class)

---

##  Data Preprocessing Pipeline

To eliminate data leakage:
1. **Train/Test Stratification**: 80% Training (1,280 samples) and 20% Testing (320 samples) stratified on `career_label`.
2. **Interest Multi-Hot Encoding**: Converts multi-select interest strings into 9 binary indicator features (`interest_data_analysis`, etc.).
3. **Categorical One-Hot Encoding**: Encodes `degree`, `specialization`, and `academic_performance` via `OneHotEncoder(handle_unknown='ignore')`.
4. **Feature Standardization**: Standardizes continuous and rating attributes using `StandardScaler`.
5. Preprocessor is fitted strictly on `X_train` and saved as `models/preprocessing.pkl`.

---

##  Machine Learning Methodology & Evaluation

Five supervised classifiers were trained and evaluated under identical stratified train/test conditions:

### Model Comparison Results:

| Model | Test Accuracy | Weighted F1-Score | Macro F1-Score | Weighted Precision | Weighted Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Classifier**  | **99.69%** | **99.69%** | **99.69%** | **99.69%** | **99.69%** |
| **Logistic Regression** | 99.06% | 99.06% | 99.06% | 99.08% | 99.06% |
| **Support Vector Machine (SVC)** | 99.06% | 99.06% | 99.06% | 99.10% | 99.06% |
| **K-Nearest Neighbors (KNN)** | 98.75% | 98.75% | 98.75% | 98.78% | 98.75% |
| **Decision Tree Classifier** | 94.69% | 94.70% | 94.70% | 94.90% | 94.69% |

*The best model (**Random Forest**) was persisted to `models/career_model.pkl`.*

---

##  Skill Gap Engine & Mathematical Formulations

For any given target career benchmark:
- Gap = Student Level - Required Level

### Category Rules:
- **Strong**: Student Level >= Required Level + 1 or (Student Level = 5 and Required Level >= 3)
- **Meets Requirement**: Student Level == Required Level
- **Needs Improvement**: Gap in {-1, -2} and Required Level > 0
- **Major Gap**: Gap <= -3 or (Required Level >= 4 and Student Level <= 1)

### Career Readiness Index:
- Technical Readiness (%) = [Sum of min(Student_i, Required_i) / Sum of Required_i] * 100
- Soft Skills Readiness (%) = [Sum of min(Student_j, Required_j) / Sum of Required_j] * 100
- Overall Career Readiness (%) = 0.70 * Technical Readiness + 0.30 * Soft Skills Readiness

---

## ? Database Design

The relational database is normalized and configured in `database/schema.sql`:

```sql
CREATE DATABASE IF NOT EXISTS career_recommendation_db;
USE career_recommendation_db;

-- 1. students: Stores demographic & academic data
-- 2. skills: Master technical and soft skill catalog
-- 3. student_skills: Student proficiency ratings (0-5)
-- 4. careers: IT career domain catalog
-- 5. career_skills: Benchmark required skill levels
-- 6. recommendations: ML predicted top-3 career matches and confidence scores
-- 7. learning_resources: Curated courses, milestones, and project links
```

*Dual-Mode Engine: Seamlessly switches to local SQLite (`data/career_recommendations.db`) when MySQL is unreachable.*

---

##  Project Structure

```
career-recommendation-system/
? app.py                         # Streamlit multi-page dashboard application
? requirements.txt               # Project dependencies
? .env.example                   # MySQL and environment configuration template
? .gitignore                     # Git ignore rules
? README.md                      # Academic documentation & viva guide
? config/
?   ? career_requirements.json   # Benchmark required skill levels (0-5)
?   ? learning_resources.json    # Curated skill courses, milestones & projects
? database/
?   ? schema.sql                 # MySQL schema DDL & seed data
? data/
?   ? raw/
?   ?   ? student_career_dataset.csv  # 1600 synthetic student records
?   ? processed/
?   ?   ? train_data.csv
?   ?   ? test_data.csv
?   ? career_recommendations.db  # SQLite local fallback database
? models/
?   ? career_model.pkl           # Persisted best Random Forest model
?   ? preprocessing.pkl          # Persisted ColumnTransformer pipeline
?   ? model_metrics.json         # Real multi-model benchmark evaluation metrics
? notebooks/
?   ? EDA_and_Model.ipynb        # Data science EDA & model training notebook
? src/
?   ? __init__.py
?   ? data_generator.py          # Synthetic dataset generator (seed=42)
?   ? preprocessing.py           # Preprocessing & validation pipeline
?   ? train_model.py             # 5-Model training & evaluation module
?   ? prediction.py              # Top-3 career prediction inference engine
?   ? skill_gap.py               # Benchmark gap analysis & readiness engine
?   ? learning_recommendation.py # Personalized 4-phase roadmap engine
?   ? database.py                # Dual-mode database manager (MySQL/SQLite)
? tests/
    ? test_pipeline.py           # Automated unit & integration test suite
```

---

##  Installation & Setup Instructions

### 1. Prerequisites
- Python 3.11 or higher
- Git
- (Optional) MySQL Server 8.0+

### 2. Clone / Open the Project
```bash
cd career-recommendation-system
```

### 3. Create & Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scriptsctivate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure MySQL (Optional)
If using MySQL, copy `.env.example` to `.env` and fill in your credentials:
```env
DB_HOST=localhost
DB_PORT=3306
DB_NAME=career_recommendation_db
DB_USER=root
DB_PASSWORD=your_mysql_password
```
Execute the database schema in your MySQL shell:
```bash
mysql -u root -p < database/schema.sql
```
*(If MySQL is not running, the application will automatically use local SQLite storage).*

---

##  Step-by-Step Execution Guide

### 1. Generate the Dataset
```bash
python src/data_generator.py
```

### 2. Train & Evaluate the Models
```bash
python src/train_model.py
```

### 3. Run Automated Tests
```bash
python tests/test_pipeline.py
```

### 4. Launch the Streamlit Web Application
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

##  Viva & Project Review Guide

### Frequently Asked Viva Questions:

**Q1: Why did you choose Random Forest as the primary model?**
> *Answer:* Random Forest is an ensemble of decision trees that handles non-linear relationships, resists overfitting via bagging (bootstrap aggregation) and feature subsampling, and achieved the highest test F1-score (99.69%) and accuracy on our stratified evaluation set.

**Q2: How does the system prevent data leakage during preprocessing?**
> *Answer:* The `CareerDataPreprocessor` is strictly fitted only on the training split (`X_train`). The fitted parameters (means, standard deviations, categorical encodings) are then applied to `X_test` and live user inputs.

**Q3: How are career match percentages calculated?**
> *Answer:* Rather than assigning arbitrary scores, the system uses the Scikit-learn model's `predict_proba()` method to compute calibrated class probabilities across all 8 career categories.

**Q4: How does the Skill Gap Engine classify skills into categories?**
> *Answer:* It calculates the difference Delta = Student Level - Required Benchmark Level. Gaps <= -3 are classified as Major Gaps, Delta in {-1, -2} as Needs Improvement, Delta = 0 as Meets Requirement, and Delta >= 1 as Strong.

**Q5: How does the dual-mode database architecture work?**
> *Answer:* `src/database.py` attempts a connection to MySQL using PyMySQL. If the MySQL server is offline or unconfigured, it seamlessly connects to an embedded SQLite database (`data/career_recommendations.db`), ensuring zero downtime and complete functionality for local demos.

---

##  Academic Disclaimer

This project was developed for academic evaluation purposes for B.Sc. Information Technology students. The synthetic dataset and benchmark skill profiles are curated for educational guidance and demonstrate full-stack machine learning engineering, data preprocessing, and web dashboard design.
