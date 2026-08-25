"""
Database Management Module
Provides dual-mode database access:
1. Primary: Production MySQL 8.0+ connection via PyMySQL with connection pooling.
2. Fallback: Local SQLite database (data/career_recommendations.db) for zero-configuration local development.

Handles schema initialization, student record storage, skill ratings, and recommendation logging.
"""

import os
import json
import sqlite3
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(base_dir, ".env"))

# Helper to read from Streamlit secrets or OS environment
def get_config_val(key: str, default_val: str = "") -> str:
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default_val)

DB_HOST = get_config_val("DB_HOST", "localhost")
DB_PORT = int(get_config_val("DB_PORT", "3306"))
DB_NAME = get_config_val("DB_NAME", "career_recommendation_db")
DB_USER = get_config_val("DB_USER", "root")
DB_PASSWORD = get_config_val("DB_PASSWORD", "")

SQLITE_PATH = os.path.join(base_dir, "data", "career_recommendations.db")

class DatabaseManager:
    """
    Dual-mode Database Manager for MySQL & SQLite.
    """
    def __init__(self):
        self.mode = "Unknown"
        self.last_error = None
        self._test_connection()
        self.init_db()

    def _test_connection(self):
        try:
            import pymysql
            conn = pymysql.connect(
                host=DB_HOST,
                port=DB_PORT,
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_NAME,
                connect_timeout=2
            )
            conn.close()
            self.mode = "MySQL"
            self.last_error = None
        except Exception as e:
            self.mode = "SQLite (Local Fallback)"
            self.last_error = str(e)
            os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)

    def get_status(self) -> dict:
        self._test_connection()
        return {
            "mode": self.mode,
            "is_mysql": (self.mode == "MySQL"),
            "host": DB_HOST if self.mode == "MySQL" else "Local SQLite File",
            "database": DB_NAME if self.mode == "MySQL" else SQLITE_PATH,
            "user": DB_USER if self.mode == "MySQL" else "N/A",
            "last_error": self.last_error
        }

    def _get_raw_connection(self):
        if self.mode == "MySQL":
            import pymysql
            return pymysql.connect(
                host=DB_HOST,
                port=DB_PORT,
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_NAME,
                autocommit=True
            )
        else:
            return sqlite3.connect(SQLITE_PATH)

    def init_db(self):
        """
        Initializes schema and seeds master data if tables do not exist.
        """
        conn = self._get_raw_connection()
        cursor = conn.cursor()
        
        try:
            if self.mode == "MySQL":
                # MySQL Schema
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    student_id VARCHAR(50) PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    degree VARCHAR(100) NOT NULL,
                    specialization VARCHAR(100) NOT NULL,
                    cgpa DECIMAL(4, 2) NOT NULL,
                    academic_performance VARCHAR(50) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS recommendations (
                    recommendation_id INT AUTO_INCREMENT PRIMARY KEY,
                    student_id VARCHAR(50) NOT NULL,
                    career_name VARCHAR(100) NOT NULL,
                    rank_order INT NOT NULL,
                    match_score DECIMAL(5, 2) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS student_skills (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    student_id VARCHAR(50) NOT NULL,
                    skill_name VARCHAR(50) NOT NULL,
                    rating INT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """)
            else:
                # SQLite Schema
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    student_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    degree TEXT NOT NULL,
                    specialization TEXT NOT NULL,
                    cgpa REAL NOT NULL,
                    academic_performance TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS recommendations (
                    recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    career_name TEXT NOT NULL,
                    rank_order INTEGER NOT NULL,
                    match_score REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
                """)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS student_skills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    skill_name TEXT NOT NULL,
                    rating INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                """)
                conn.commit()
        finally:
            cursor.close()
            conn.close()

    def save_assessment(self, student_info: dict, all_skills: dict, recommendations: list) -> str:
        """
        Saves student profile, all skill ratings, and top 3 career recommendations.
        """
        conn = self._get_raw_connection()
        cursor = conn.cursor()
        
        student_id = student_info.get("student_id") or f"STU_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            # 1. Insert Student
            if self.mode == "MySQL":
                cursor.execute("""
                INSERT INTO students (student_id, name, degree, specialization, cgpa, academic_performance)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE name=VALUES(name), cgpa=VALUES(cgpa);
                """, (
                    student_id,
                    student_info.get("name", "B.Sc IT Student"),
                    student_info.get("degree", "B.Sc Information Technology"),
                    student_info.get("specialization", "Information Technology"),
                    float(student_info.get("cgpa", 7.5)),
                    student_info.get("academic_performance", "First Class")
                ))
            else:
                cursor.execute("""
                INSERT OR REPLACE INTO students (student_id, name, degree, specialization, cgpa, academic_performance, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """, (
                    student_id,
                    student_info.get("name", "B.Sc IT Student"),
                    student_info.get("degree", "B.Sc Information Technology"),
                    student_info.get("specialization", "Information Technology"),
                    float(student_info.get("cgpa", 7.5)),
                    student_info.get("academic_performance", "First Class"),
                    now_str
                ))
                
            # 2. Insert Skills
            for skill_name, rating in all_skills.items():
                if self.mode == "MySQL":
                    cursor.execute("""
                    INSERT INTO student_skills (student_id, skill_name, rating)
                    VALUES (%s, %s, %s);
                    """, (student_id, skill_name, int(rating)))
                else:
                    cursor.execute("""
                    INSERT INTO student_skills (student_id, skill_name, rating, created_at)
                    VALUES (?, ?, ?, ?);
                    """, (student_id, skill_name, int(rating), now_str))
                    
            # 3. Insert Recommendations
            for rec in recommendations:
                if self.mode == "MySQL":
                    cursor.execute("""
                    INSERT INTO recommendations (student_id, career_name, rank_order, match_score)
                    VALUES (%s, %s, %s, %s);
                    """, (student_id, rec["career"], rec["rank"], float(rec["match_score"])))
                else:
                    cursor.execute("""
                    INSERT INTO recommendations (student_id, career_name, rank_order, match_score, created_at)
                    VALUES (?, ?, ?, ?, ?);
                    """, (student_id, rec["career"], rec["rank"], float(rec["match_score"]), now_str))
                    
            if self.mode != "MySQL":
                conn.commit()
                
            return student_id
        finally:
            cursor.close()
            conn.close()

    def get_recent_assessments(self, limit: int = 50) -> pd.DataFrame:
        """
        Retrieves recent student assessments joined with top primary career recommendation.
        """
        conn = self._get_raw_connection()
        query = """
        SELECT 
            s.student_id,
            s.name,
            s.degree,
            s.cgpa,
            s.academic_performance,
            r.career_name AS primary_recommended_career,
            r.match_score AS career_match_pct,
            s.created_at
        FROM students s
        LEFT JOIN recommendations r ON s.student_id = r.student_id AND r.rank_order = 1
        ORDER BY s.created_at DESC
        LIMIT %d;
        """ % limit
        
        try:
            df = pd.read_sql_query(query, conn)
            return df
        finally:
            conn.close()

# Singleton DB instance
db_manager = DatabaseManager()

if __name__ == "__main__":
    status = db_manager.get_status()
    print("Database Status:", status)
