-- Database Schema for Career Recommendation and Skill Gap Analysis System
-- Target Database: MySQL 8.0+
-- Database Name: career_recommendation_db

CREATE DATABASE IF NOT EXISTS career_recommendation_db;
USE career_recommendation_db;

-- 1. Students Table
CREATE TABLE IF NOT EXISTS students (
    student_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    degree VARCHAR(100) NOT NULL DEFAULT 'B.Sc Information Technology',
    specialization VARCHAR(100) NOT NULL DEFAULT 'Information Technology',
    cgpa DECIMAL(4, 2) NOT NULL,
    academic_performance VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 2. Skills Master Table
CREATE TABLE IF NOT EXISTS skills (
    skill_id INT AUTO_INCREMENT PRIMARY KEY,
    skill_name VARCHAR(50) NOT NULL UNIQUE,
    category ENUM('Technical', 'Soft') NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 3. Student Skills Ratings Table
CREATE TABLE IF NOT EXISTS student_skills (
    id INT AUTO_INCREMENT PRIMARY KEY,
    student_id VARCHAR(50) NOT NULL,
    skill_id INT NOT NULL,
    rating INT NOT NULL CHECK (rating BETWEEN 0 AND 5),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
    FOREIGN KEY (skill_id) REFERENCES skills(skill_id) ON DELETE CASCADE,
    UNIQUE KEY unique_student_skill (student_id, skill_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 4. Careers Master Table
CREATE TABLE IF NOT EXISTS careers (
    career_id INT AUTO_INCREMENT PRIMARY KEY,
    career_name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT NOT NULL,
    required_cgpa DECIMAL(4, 2) DEFAULT 6.50
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 5. Career Skill Benchmark Requirements Table
CREATE TABLE IF NOT EXISTS career_skills (
    id INT AUTO_INCREMENT PRIMARY KEY,
    career_id INT NOT NULL,
    skill_id INT NOT NULL,
    required_level INT NOT NULL CHECK (required_level BETWEEN 0 AND 5),
    FOREIGN KEY (career_id) REFERENCES careers(career_id) ON DELETE CASCADE,
    FOREIGN KEY (skill_id) REFERENCES skills(skill_id) ON DELETE CASCADE,
    UNIQUE KEY unique_career_skill (career_id, skill_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 6. ML Model Recommendations Table
CREATE TABLE IF NOT EXISTS recommendations (
    recommendation_id INT AUTO_INCREMENT PRIMARY KEY,
    student_id VARCHAR(50) NOT NULL,
    career_id INT NOT NULL,
    rank_order INT NOT NULL CHECK (rank_order BETWEEN 1 AND 3),
    match_score DECIMAL(5, 2) NOT NULL,
    model_version VARCHAR(50) DEFAULT 'RandomForest_v1.0',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
    FOREIGN KEY (career_id) REFERENCES careers(career_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 7. Learning Resources Master Table
CREATE TABLE IF NOT EXISTS learning_resources (
    resource_id INT AUTO_INCREMENT PRIMARY KEY,
    skill_id INT NOT NULL,
    resource_name VARCHAR(200) NOT NULL,
    resource_type ENUM('Course', 'Certification', 'Video Series', 'Book', 'Practice Platform', 'Project') NOT NULL,
    url VARCHAR(500),
    description TEXT,
    difficulty ENUM('Beginner', 'Intermediate', 'Advanced') DEFAULT 'Beginner',
    FOREIGN KEY (skill_id) REFERENCES skills(skill_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ====================================================================
-- SEED DATA
-- ====================================================================

INSERT IGNORE INTO skills (skill_name, category) VALUES
('Python', 'Technical'),
('Java', 'Technical'),
('C_CPP', 'Technical'),
('SQL', 'Technical'),
('HTML_CSS', 'Technical'),
('JavaScript', 'Technical'),
('Excel', 'Technical'),
('Power_BI', 'Technical'),
('Statistics', 'Technical'),
('Machine_Learning', 'Technical'),
('Deep_Learning', 'Technical'),
('Cloud_Computing', 'Technical'),
('Networking', 'Technical'),
('Cybersecurity', 'Technical'),
('Git_GitHub', 'Technical'),
('Communication', 'Soft'),
('Problem_Solving', 'Soft'),
('Teamwork', 'Soft'),
('Leadership', 'Soft'),
('Analytical_Thinking', 'Soft');

INSERT IGNORE INTO careers (career_name, description, required_cgpa) VALUES
('Data Analyst', 'Transforms raw data into actionable insights, dashboards, and business reports using statistical techniques and BI tools.', 6.50),
('Data Scientist', 'Builds predictive statistical models, analyzes complex datasets, and extracts strategic business value using advanced ML algorithms.', 7.50),
('Software Developer', 'Designs, writes, tests, and maintains robust software applications, backend services, and scalable desktop or mobile systems.', 6.50),
('Web Developer', 'Builds responsive, user-friendly frontend interfaces and powerful server-side web applications and APIs.', 6.00),
('Database Administrator', 'Ensures database performance, integrity, security, backup, recovery, and seamless query optimization for enterprise data.', 6.50),
('Cloud Engineer', 'Architects, deploys, scales, and manages reliable cloud infrastructure, serverless architectures, and CI/CD pipelines.', 6.80),
('Cybersecurity Analyst', 'Protects organizational networks, systems, and sensitive data from cyber threats, vulnerabilities, and unauthorized breaches.', 6.80),
('Machine Learning Engineer', 'Bridges data science and software engineering by building, optimizing, containerizing, and deploying scalable ML and DL models to production.', 7.50);
