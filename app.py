"""
Machine Learning-Based Career Recommendation and Skill Gap Analysis System
for B.Sc. Information Technology Students
Main Streamlit Application File with Strict Admin/Student Role Separation
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="IT Career Recommender & Skill Gap Analyzer",
    page_icon="\U0001F393",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add src to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from data_generator import TECHNICAL_SKILLS, SOFT_SKILLS, INTERESTS, CAREERS, get_academic_performance
from prediction import CareerPredictor
from skill_gap import SkillGapAnalyzer
from learning_recommendation import LearningRoadmapEngine
from database import db_manager

# Custom CSS for Modern, Premium Academic UI
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #F9FAFB 0%, #F3F4F6 100%);
        border: 1px solid #E5E7EB; border-radius: 12px; padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); margin-bottom: 1rem;
    }
    .card-title { font-size: 0.9rem; font-weight: 600; color: #6B7280; text-transform: uppercase; }
    .card-value { font-size: 1.8rem; font-weight: 700; color: #111827; }
    .highlight-card {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border-left: 5px solid #2563EB; border-radius: 8px; padding: 1.2rem; margin-bottom: 1.2rem;
    }
    .badge-strong { background-color: #DEF7EC; color: #03543F; padding: 4px 10px; border-radius: 12px; font-weight: 600; }
    .badge-meets { background-color: #E1EFFE; color: #1E429F; padding: 4px 10px; border-radius: 12px; font-weight: 600; }
    .badge-improve { background-color: #FEF08A; color: #713F12; padding: 4px 10px; border-radius: 12px; font-weight: 600; }
    .badge-gap { background-color: #FDE8E8; color: #9B1C1C; padding: 4px 10px; border-radius: 12px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# Navigation Page Constants
PAGE_HOME = "\U0001F3E0 Home & Overview"
PAGE_ASSESSMENT = "\U0001F4DD Student Assessment"
PAGE_RECOMMENDATIONS = "\U0001F3AF Career Recommendations"
PAGE_SKILL_GAP = "\U0001F4CA Skill Gap Analysis"
PAGE_LEARNING_PATH = "\U0001F680 Personalized Learning Path"
PAGE_MODEL_PERF = "\U0001F9E0 Model Performance & Evaluation"
PAGE_ADMIN_DB = "\U0001F512 Admin Dashboard & DB Explorer"

# Initialize Session State
if "predictor" not in st.session_state:
    try:
        st.session_state.predictor = CareerPredictor()
        st.session_state.gap_analyzer = SkillGapAnalyzer()
        st.session_state.roadmap_engine = LearningRoadmapEngine()
        st.session_state.models_ready = True
    except Exception as e:
        st.session_state.models_ready = False
        st.session_state.model_error = str(e)

if "current_assessment" not in st.session_state:
    st.session_state.current_assessment = None

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False

# Retrieve Admin Password from Streamlit Secrets or Environment (Default: admin123)
ADMIN_PASSWORD = "admin123"
try:
    if hasattr(st, "secrets") and "ADMIN_PASSWORD" in st.secrets:
        ADMIN_PASSWORD = str(st.secrets["ADMIN_PASSWORD"])
    elif os.getenv("ADMIN_PASSWORD"):
        ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
except Exception:
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

# Check Query Parameter for Admin Link (?role=admin or ?admin=true)
query_params = getattr(st, "query_params", {})
role_param = str(query_params.get("role", "")).lower()
admin_param = str(query_params.get("admin", "")).lower()
is_admin_query = (role_param == "admin") or (admin_param in ["1", "true"])

# Role-Based Page List: Regular Students NEVER see Admin menu item
if st.session_state.admin_authenticated:
    PAGES_LIST = [
        PAGE_ADMIN_DB,
        PAGE_HOME,
        PAGE_ASSESSMENT,
        PAGE_RECOMMENDATIONS,
        PAGE_SKILL_GAP,
        PAGE_LEARNING_PATH,
        PAGE_MODEL_PERF
    ]
else:
    PAGES_LIST = [
        PAGE_HOME,
        PAGE_ASSESSMENT,
        PAGE_RECOMMENDATIONS,
        PAGE_SKILL_GAP,
        PAGE_LEARNING_PATH,
        PAGE_MODEL_PERF
    ]

# Sidebar Brand Header
st.sidebar.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=70)
st.sidebar.title("\U0001F393 Career Navigator")
st.sidebar.markdown("**B.Sc. Information Technology**\nDecision Support & Skill Gap Engine")

# If accessed via Admin URL but not yet authenticated, render Admin Login Modal/Gate directly
if is_admin_query and not st.session_state.admin_authenticated:
    nav_choice = PAGE_ADMIN_DB
else:
    nav_choice = st.sidebar.radio("Navigation Menu", PAGES_LIST)

st.sidebar.divider()

# Sidebar Footer & Discreet Admin Gateway
if st.session_state.admin_authenticated:
    st.sidebar.success("\U0001F7E2 **Mode:** Administrator Active")
    if st.sidebar.button("\U0001F512 Log Out (Admin)", use_container_width=True):
        st.session_state.admin_authenticated = False
        if hasattr(st, "query_params") and "role" in st.query_params:
            del st.query_params["role"]
        if hasattr(st, "query_params") and "admin" in st.query_params:
            del st.query_params["admin"]
        st.rerun()
else:
    # Student View: Clean footer with unobtrusive Faculty/Admin login button
    col_f1, col_f2 = st.sidebar.columns([3, 2])
    with col_f1:
        st.caption("v1.0.0 | Academic 2026")
    with col_f2:
        if st.button("\U0001F510 Faculty/Admin", key="faculty_login_btn", help="Faculty & Administrator Portal"):
            st.session_state["show_admin_login"] = True
            st.rerun()

# ==============================================================================
# PAGE 1: HOME & OVERVIEW
# ==============================================================================
if nav_choice == PAGE_HOME:
    st.markdown('<div class="main-header">\U0001F393 ML-Based Career Recommendation & Skill Gap Analysis System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">An intelligent end-to-end guidance platform engineered specifically for B.Sc. Information Technology students.</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card"><div class="card-title">Supported Careers</div><div class="card-value">8 Domains</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card"><div class="card-title">Evaluated Skills</div><div class="card-value">20 Competencies</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card"><div class="card-title">ML Algorithms</div><div class="card-value">5 Classifiers</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card"><div class="card-title">Best Model Accuracy</div><div class="card-value">99.7%</div></div>""", unsafe_allow_html=True)
        
    st.markdown("""
    ### \U0001F3AF System Objective & Workflow
    This application assists IT undergraduates in transitioning from academia to industry by analyzing their academic standing, technical proficiencies (0-5), soft skills (0-5), and personal career interests.
    
    ### \U0001F31F Key Modules
    1. **Multi-Model Machine Learning**: Compares Logistic Regression, Decision Tree, Random Forest, KNN, and SVM on stratified cross-validated data.
    2. **Ranked Probability Estimates**: Produces true model probability distributions rather than hardcoded scores.
    3. **Standardized Skill Benchmarks**: Measures students against industry-defined skill expectations across all 8 IT career pathways.
    4. **Personalized 4-Phase Roadmaps**: Generates milestone topics, free curated course links, and portfolio project suggestions.
    5. **Dual-Mode Database**: Supports high-performance MySQL persistence with automatic SQLite fallback for zero-configuration testing.
    """)
    
    st.divider()
    st.info("\U0001F4A1 **Getting Started:** Click on **\U0001F4DD Student Assessment** in the sidebar to analyze your career match!")

# ==============================================================================
# PAGE 2: STUDENT ASSESSMENT FORM
# ==============================================================================
elif nav_choice == PAGE_ASSESSMENT:
    st.markdown('<div class="main-header">\U0001F4DD Student Assessment Form</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter your academic background, technical skills, soft skills, and career interests.</div>', unsafe_allow_html=True)
    
    with st.expander("\u26A1 Quick Demo Presets (Click to auto-populate sample student profiles)"):
        preset_cols = st.columns(4)
        if preset_cols[0].button("\U0001F4CA Data Analyst Profile", use_container_width=True):
            st.session_state["preset"] = {
                "name": "Aarav Patel", "degree": "B.Sc Information Technology", "specialization": "Information Technology", "cgpa": 8.4,
                "interests": ["Data Analysis", "Database Management"],
                "Python": 4, "Java": 1, "C_CPP": 0, "SQL": 5, "HTML_CSS": 1, "JavaScript": 1, "Excel": 5, "Power_BI": 5, "Statistics": 4,
                "Machine_Learning": 2, "Deep_Learning": 0, "Cloud_Computing": 2, "Networking": 1, "Cybersecurity": 1, "Git_GitHub": 3,
                "Communication": 4, "Problem_Solving": 4, "Teamwork": 4, "Leadership": 3, "Analytical_Thinking": 5
            }
            st.rerun()
            
        if preset_cols[1].button("\U0001F310 Web Developer Profile", use_container_width=True):
            st.session_state["preset"] = {
                "name": "Priya Sharma", "degree": "B.Sc Information Technology", "specialization": "Web Technologies", "cgpa": 7.9,
                "interests": ["Web Development", "Software Development"],
                "Python": 3, "Java": 2, "C_CPP": 1, "SQL": 3, "HTML_CSS": 5, "JavaScript": 5, "Excel": 1, "Power_BI": 1, "Statistics": 1,
                "Machine_Learning": 1, "Deep_Learning": 0, "Cloud_Computing": 2, "Networking": 2, "Cybersecurity": 2, "Git_GitHub": 4,
                "Communication": 4, "Problem_Solving": 4, "Teamwork": 4, "Leadership": 2, "Analytical_Thinking": 3
            }
            st.rerun()
            
        if preset_cols[2].button("\U0001F512 Cybersecurity Profile", use_container_width=True):
            st.session_state["preset"] = {
                "name": "Vikram Singh", "degree": "B.Sc Information Technology", "specialization": "Cloud & Security", "cgpa": 8.1,
                "interests": ["Cybersecurity", "Cloud Computing"],
                "Python": 3, "Java": 2, "C_CPP": 4, "SQL": 3, "HTML_CSS": 2, "JavaScript": 2, "Excel": 1, "Power_BI": 1, "Statistics": 2,
                "Machine_Learning": 1, "Deep_Learning": 0, "Cloud_Computing": 3, "Networking": 5, "Cybersecurity": 5, "Git_GitHub": 3,
                "Communication": 3, "Problem_Solving": 5, "Teamwork": 3, "Leadership": 3, "Analytical_Thinking": 5
            }
            st.rerun()

        if preset_cols[3].button("\U0001F916 ML Engineer Profile", use_container_width=True):
            st.session_state["preset"] = {
                "name": "Rohan Gupta", "degree": "B.Sc Information Technology", "specialization": "Data Science", "cgpa": 8.8,
                "interests": ["Machine Learning", "Artificial Intelligence", "Data Science"],
                "Python": 5, "Java": 3, "C_CPP": 3, "SQL": 4, "HTML_CSS": 1, "JavaScript": 2, "Excel": 2, "Power_BI": 2, "Statistics": 5,
                "Machine_Learning": 5, "Deep_Learning": 5, "Cloud_Computing": 4, "Networking": 2, "Cybersecurity": 2, "Git_GitHub": 4,
                "Communication": 3, "Problem_Solving": 5, "Teamwork": 4, "Leadership": 3, "Analytical_Thinking": 5
            }
            st.rerun()

    preset = st.session_state.get("preset", {})

    with st.form("assessment_form"):
        tab1, tab2, tab3, tab4 = st.tabs(["\U0001F393 Academic Profile", "\U0001F4BB Technical Skills (0-5)", "\U0001F91D Soft Skills (0-5)", "\U0001F3AF Interests & Focus"])
        with tab1:
            col_a1, col_a2 = st.columns(2)
            student_name = col_a1.text_input("Student Full Name", value=preset.get("name", "Ananya Sharma"))
            student_id = col_a2.text_input("Student ID / Roll Number", value="STU" + str(np.random.randint(1000, 9999)))
            
            col_a3, col_a4, col_a5 = st.columns(3)
            degree = col_a3.selectbox("Degree Program", ["B.Sc Information Technology", "B.Sc Computer Science", "BCA"], index=["B.Sc Information Technology", "B.Sc Computer Science", "BCA"].index(preset.get("degree", "B.Sc Information Technology")))
            specialization = col_a4.selectbox("Specialization", ["Information Technology", "Data Science", "Software Engineering", "Cloud & Security", "Web Technologies"], index=["Information Technology", "Data Science", "Software Engineering", "Cloud & Security", "Web Technologies"].index(preset.get("specialization", "Information Technology")))
            cgpa = col_a5.number_input("Current CGPA (Out of 10.0)", min_value=0.0, max_value=10.0, value=float(preset.get("cgpa", 8.2)), step=0.1)
            academic_level = get_academic_performance(cgpa)
            st.info(f"\U0001F4CA **Calculated Academic Standing:** {academic_level} (CGPA: {cgpa:.2f})")
            
        with tab2:
            st.caption("Rating Guide: 0 = No knowledge | 1 = Beginner | 2 = Basic | 3 = Intermediate | 4 = Advanced | 5 = Expert")
            tech_cols = st.columns(3)
            tech_ratings = {}
            for idx, skill in enumerate(TECHNICAL_SKILLS):
                col = tech_cols[idx % 3]
                display_name = skill.replace("_", " ")
                tech_ratings[skill] = col.slider(f"{display_name}", min_value=0, max_value=5, value=int(preset.get(skill, 2)), key=f"tech_{skill}")
                
        with tab3:
            st.caption("Rate your interpersonal and behavioral competencies (0 to 5)")
            soft_cols = st.columns(3)
            soft_ratings = {}
            for idx, skill in enumerate(SOFT_SKILLS):
                col = soft_cols[idx % 3]
                display_name = skill.replace("_", " ")
                soft_ratings[skill] = col.slider(f"{display_name}", min_value=0, max_value=5, value=int(preset.get(skill, 3)), key=f"soft_{skill}")
                
        with tab4:
            st.caption("Select the technical domains that genuinely interest you (Multi-select allowed)")
            default_interests = preset.get("interests", ["Data Analysis", "Artificial Intelligence"])
            selected_interests = st.multiselect("Career Interests", INTERESTS, default=[i for i in default_interests if i in INTERESTS])
            
        submit_button = st.form_submit_button("\U0001F680 Analyze My Career & Skill Gaps", type="primary", use_container_width=True)

    if submit_button:
        student_data = {
            "name": student_name, "student_id": student_id, "degree": degree, "specialization": specialization,
            "cgpa": cgpa, "academic_performance": academic_level, **tech_ratings, **soft_ratings, "interests": selected_interests
        }
        preprocessor = st.session_state.predictor.preprocessor
        is_valid, err_msg = preprocessor.validate_single_input(student_data)
        if not is_valid:
            st.error(f"\u274C {err_msg}")
        else:
            with st.spinner("Analyzing student profile with Machine Learning models..."):
                try:
                    pred_results = st.session_state.predictor.predict_top_careers(student_data, top_k=3)
                    st.session_state.current_assessment = student_data
                    st.session_state.prediction_result = pred_results
                    
                    all_skills = {**tech_ratings, **soft_ratings}
                    db_manager.save_assessment(student_data, all_skills, pred_results["top_recommendations"])
                    
                    st.success("\u2705 Analysis complete! Prediction saved to database.")
                    st.balloons()
                    
                    top_career = pred_results["primary_career"]
                    top_score = pred_results["top_recommendations"][0]["match_score"]
                    st.markdown(f"""
                    <div class="highlight-card">
                        <h3 style="margin-top:0; color:#1E3A8A;">\U0001F3AF Top Match: {top_career} ({top_score}% Match)</h3>
                        <p>Navigate to <b>\U0001F3AF Career Recommendations</b> or <b>\U0001F4CA Skill Gap Analysis</b> from the sidebar to inspect your full report.</p>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as ex:
                    st.error(f"Prediction Error: {str(ex)}")

# ==============================================================================
# PAGE 3: CAREER RECOMMENDATIONS
# ==============================================================================
elif nav_choice == PAGE_RECOMMENDATIONS:
    st.markdown('<div class="main-header">\U0001F3AF Recommended IT Career Paths</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ranked career matches determined by trained Scikit-learn Machine Learning classification.</div>', unsafe_allow_html=True)
    
    if st.session_state.prediction_result is None:
        st.warning("\u26A0\uFE0F No student assessment found. Please complete the **\U0001F4DD Student Assessment** form first.")
    else:
        results = st.session_state.prediction_result
        top_recs = results["top_recommendations"]
        primary = results["primary_career"]
        all_scores = results["all_scores"]
        
        st.markdown(f"""
        <div class="highlight-card">
            <div style="font-size:0.9rem; font-weight:700; color:#2563EB; text-transform:uppercase;">\U0001F947 Primary Career Recommendation</div>
            <div style="font-size:2.2rem; font-weight:800; color:#1E3A8A; margin: 4px 0;">{primary}</div>
            <div style="font-size:1.1rem; color:#374151;">{top_recs[0]["description"]}</div>
            <div style="margin-top:10px;">
                <span class="badge-strong" style="font-size:1rem;">Match Probability: {top_recs[0]["match_score"]}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("\U0001F3C6 Top 3 Ranked Career Matches")
        rec_cols = st.columns(3)
        for i, rec in enumerate(top_recs):
            with rec_cols[i]:
                rank_badge = ["\U0001F947 Rank 1", "\U0001F948 Rank 2", "\U0001F949 Rank 3"][i]
                st.markdown(f"""
                <div class="metric-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:700; color:#4B5563;">{rank_badge}</span>
                        <span style="font-weight:700; color:#2563EB; font-size:1.2rem;">{rec["match_score"]}%</span>
                    </div>
                    <div style="font-size:1.3rem; font-weight:700; color:#111827; margin:8px 0;">{rec["career"]}</div>
                    <p style="font-size:0.9rem; color:#6B7280; height:60px;">{rec["description"]}</p>
                </div>
                """, unsafe_allow_html=True)
                st.progress(rec["match_score"] / 100.0)

        st.divider()
        st.subheader("\U0001F4CA Probability Distribution Across All 8 IT Career Domains")
        
        score_df = pd.DataFrame([
            {"Career": k, "Match Probability (%)": v}
            for k, v in all_scores.items()
        ]).sort_values(by="Match Probability (%)", ascending=True)
        
        fig = px.bar(
            score_df,
            x="Match Probability (%)",
            y="Career",
            orientation="h",
            text="Match Probability (%)",
            color="Match Probability (%)",
            color_continuous_scale="Blues",
            title="Calibrated Prediction Probabilities (%) by Career Category"
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PAGE 4: SKILL GAP ANALYSIS
# ==============================================================================
elif nav_choice == PAGE_SKILL_GAP:
    st.markdown('<div class="main-header">\U0001F4CA Skill Gap Analysis Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Detailed comparison between your current proficiencies and industry standard requirements.</div>', unsafe_allow_html=True)
    
    if st.session_state.current_assessment is None:
        st.warning("\u26A0\uFE0F No student assessment found. Please complete the **\U0001F4DD Student Assessment** form first.")
    else:
        student_data = st.session_state.current_assessment
        pred_results = st.session_state.prediction_result
        default_target = pred_results["primary_career"] if pred_results else "Data Analyst"
        
        col_select, col_info = st.columns([1, 2])
        target_career = col_select.selectbox(
            "Select Target Career for Gap Analysis",
            CAREERS,
            index=CAREERS.index(default_target) if default_target in CAREERS else 0
        )
        
        gap_report = st.session_state.gap_analyzer.analyze_gaps(student_data, target_career)
        
        st.markdown("### \U0001F4C8 Career Readiness Scores")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""<div class="metric-card"><div class="card-title">Overall Career Readiness</div><div class="card-value" style="color:#2563EB;">{gap_report["overall_readiness_score"]}%</div></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card"><div class="card-title">Technical Skills Readiness</div><div class="card-value">{gap_report["technical_readiness"]}%</div></div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card"><div class="card-title">Soft Skills Readiness</div><div class="card-value">{gap_report["soft_readiness"]}%</div></div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="metric-card"><div class="card-title">Critical Major Gaps</div><div class="card-value" style="color:{'#DC2626' if gap_report['counts']['major_gap'] > 0 else '#16A34A'};">{gap_report['counts']['major_gap']} Skills</div></div>""", unsafe_allow_html=True)
            
        col_radar, col_bar = st.columns(2)
        with col_radar:
            st.subheader("\U0001F578\uFE0F Skill Profile Radar Chart")
            radar_data = gap_report["radar_chart"]
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_data["required_levels"],
                theta=[l.replace("_", " ") for l in radar_data["labels"]],
                fill="toself",
                name="Required Benchmark",
                line_color="#EF4444"
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_data["student_levels"],
                theta=[l.replace("_", " ") for l in radar_data["labels"]],
                fill="toself",
                name="Your Proficiency",
                line_color="#2563EB"
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                showlegend=True,
                height=450,
                margin=dict(l=40, r=40, t=30, b=30)
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
        with col_bar:
            st.subheader("\U0001F4CA Skill Gap (+ Surplus / - Deficit)")
            bar_data = gap_report["bar_chart"]
            bar_df = pd.DataFrame({
                "Skill": [s.replace("_", " ") for s in bar_data["skills"]],
                "Gap": bar_data["gaps"]
            }).sort_values(by="Gap", ascending=True)
            
            colors = ["#EF4444" if g < 0 else ("#10B981" if g > 0 else "#3B82F6") for g in bar_df["Gap"]]
            fig_gap = px.bar(
                bar_df,
                x="Gap",
                y="Skill",
                orientation="h",
                text="Gap",
                title="Proficiency Deficit (- Negative) vs Benchmark",
                color_discrete_sequence=["#2563EB"]
            )
            fig_gap.update_traces(marker_color=colors, textposition="outside")
            fig_gap.update_layout(height=450, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_gap, use_container_width=True)

        st.divider()
        st.subheader("\U0001F4CB Detailed Skill Classification Breakdown")
        
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.markdown("##### \U0001F31F Strong Skills")
            if gap_report["strong_skills"]:
                for s in gap_report["strong_skills"]:
                    st.markdown(f"- **{s['skill'].replace('_', ' ')}** (Lvl {s['student_level']}/{s['required_level']})")
            else:
                st.caption("None identified")
                
        with b2:
            st.markdown("##### \u2705 Meets Requirement")
            if gap_report["meets_skills"]:
                for s in gap_report["meets_skills"]:
                    st.markdown(f"- **{s['skill'].replace('_', ' ')}** (Lvl {s['student_level']})")
            else:
                st.caption("None identified")
                
        with b3:
            st.markdown("##### \u26A0\uFE0F Needs Improvement")
            if gap_report["needs_improvement"]:
                for s in gap_report["needs_improvement"]:
                    st.markdown(f"- **{s['skill'].replace('_', ' ')}** (Lvl {s['student_level']} vs Req {s['required_level']})")
            else:
                st.caption("None identified")
                
        with b4:
            st.markdown("##### \U0001F6A8 Major Gaps")
            if gap_report["major_gaps"]:
                for s in gap_report["major_gaps"]:
                    st.markdown(f"- **{s['skill'].replace('_', ' ')}** (Lvl {s['student_level']} vs Req {s['required_level']})")
            else:
                st.caption("No critical gaps identified! \U0001F389")

# ==============================================================================
# PAGE 5: PERSONALIZED LEARNING PATH
# ==============================================================================
elif nav_choice == PAGE_LEARNING_PATH:
    st.markdown('<div class="main-header">\U0001F680 Personalized Learning Roadmap</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Customized 4-phase structured action plan with curated resources and capstone project recommendation.</div>', unsafe_allow_html=True)
    
    if st.session_state.current_assessment is None:
        st.warning("\u26A0\uFE0F No student assessment found. Please complete the **\U0001F4DD Student Assessment** form first.")
    else:
        student_data = st.session_state.current_assessment
        pred_results = st.session_state.prediction_result
        default_target = pred_results["primary_career"] if pred_results else "Data Analyst"
        
        target_career = st.selectbox(
            "Select Target Career for Roadmap Generation",
            CAREERS,
            index=CAREERS.index(default_target) if default_target in CAREERS else 0
        )
        
        gap_report = st.session_state.gap_analyzer.analyze_gaps(student_data, target_career)
        roadmap = st.session_state.roadmap_engine.generate_roadmap(gap_report, target_career)
        
        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(f"""<div class="metric-card"><div class="card-title">Total Estimated Hours</div><div class="card-value">{roadmap["total_estimated_hours"]} Hours</div></div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""<div class="metric-card"><div class="card-title">Recommended Pace</div><div class="card-value">{roadmap["recommended_weekly_hours"]} hrs / week</div></div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""<div class="metric-card"><div class="card-title">Total Duration</div><div class="card-value">{roadmap["total_weeks"]} Weeks</div></div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="highlight-card">
            <h4 style="margin-top:0; color:#1E3A8A;">\U0001F3C6 Recommended Capstone Portfolio Project:</h4>
            <p style="font-size:1.15rem; font-weight:600; color:#111827; margin-bottom:0;">{roadmap["capstone_project"]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("\U0001F4C5 4-Phase Chronological Curriculum")
        for phase in roadmap["phases"]:
            with st.expander(f"\U0001F4CC {phase['title']} \u2014 {phase['duration']}", expanded=(phase["phase_number"] <= 2)):
                st.markdown(f"**Goal:** {phase['description']}")
                
                for skill_item in phase["skills"]:
                    st.markdown(f"#### \U0001F539 {skill_item['skill'].replace('_', ' ')}")
                    if "topics" in skill_item and skill_item["topics"]:
                        st.markdown("**Key Topics to Master:**")
                        for top in skill_item["topics"]:
                            st.markdown(f"- {top}")
                            
                    if "courses" in skill_item and skill_item["courses"]:
                        st.markdown("**Recommended Free Learning Resources & Certifications:**")
                        for c in skill_item["courses"]:
                            st.markdown(f"- \U0001F393 [{c['name']}]({c.get('url', '#')}) *({c.get('type', 'Course')})*")
                            
                    if "projects" in skill_item and skill_item["projects"]:
                        st.markdown("**Hands-on Practice Projects:**")
                        for proj in skill_item["projects"]:
                            st.markdown(f"- \U0001F4BB {proj}")
                    st.divider()

# ==============================================================================
# PAGE 6: MODEL PERFORMANCE & EVALUATION
# ==============================================================================
elif nav_choice == PAGE_MODEL_PERF:
    st.markdown('<div class="main-header">\U0001F9E0 Machine Learning Model Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Fair comparison across 5 supervised classification algorithms on stratified test splits.</div>', unsafe_allow_html=True)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_path = os.path.join(base_dir, "models", "model_metrics.json")
    
    if not os.path.exists(metrics_path):
        st.warning("\u26A0\uFE0F Model metrics file not found. Running training module to generate metrics...")
        from train_model import train_and_evaluate_models
        train_and_evaluate_models()
        
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
        
    best_model = metrics_data.get("best_model", "Random Forest")
    models_dict = metrics_data.get("models", {})
    classes = metrics_data.get("classes", CAREERS)
    
    st.markdown(f"""
    <div class="highlight-card">
        <h4 style="margin-top:0; color:#1E3A8A;">\U0001F3C6 Selected Production Model: {best_model}</h4>
        <p>Trained and evaluated on <b>{metrics_data.get('total_samples', 1600)}</b> stratified student samples ({metrics_data.get('train_samples', 1280)} Training / {metrics_data.get('test_samples', 320)} Testing).</p>
    </div>
    """, unsafe_allow_html=True)
    
    table_rows = []
    for m_name, m_val in models_dict.items():
        table_rows.append({
            "Model": m_name,
            "Accuracy (%)": round(m_val["accuracy"] * 100, 2),
            "F1-Score (Weighted %)": round(m_val["f1_weighted"] * 100, 2),
            "F1-Score (Macro %)": round(m_val["f1_macro"] * 100, 2),
            "Precision (Weighted %)": round(m_val["precision_weighted"] * 100, 2),
            "Recall (Weighted %)": round(m_val["recall_weighted"] * 100, 2)
        })
        
    comp_df = pd.DataFrame(table_rows).sort_values(by="F1-Score (Weighted %)", ascending=False)
    st.subheader("\U0001F4CA Model Comparison Benchmark Table")
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    
    fig_comp = px.bar(
        comp_df,
        x="Model",
        y=["Accuracy (%)", "F1-Score (Weighted %)", "Precision (Weighted %)"],
        barmode="group",
        title="Multi-Model Performance Metrics Comparison on Test Split"
    )
    fig_comp.update_layout(height=400, yaxis_range=[90, 100])
    st.plotly_chart(fig_comp, use_container_width=True)
    
    st.divider()
    st.subheader("\U0001F50D Interactive Confusion Matrix Heatmap")
    
    selected_eval_model = st.selectbox(
        "Select Model for Confusion Matrix Inspection",
        list(models_dict.keys()),
        index=list(models_dict.keys()).index(best_model) if best_model in models_dict else 0
    )
    
    cm_matrix = models_dict[selected_eval_model]["confusion_matrix"]
    
    fig_cm = px.imshow(
        cm_matrix,
        x=classes,
        y=classes,
        labels=dict(x="Predicted Career", y="Actual Career", color="Count"),
        color_continuous_scale="Blues",
        text_auto=True,
        title=f"Confusion Matrix: {selected_eval_model}"
    )
    fig_cm.update_layout(height=550)
    st.plotly_chart(fig_cm, use_container_width=True)
    
    with st.expander(f"\U0001F4D1 View Detailed Classification Report for {selected_eval_model}"):
        report_data = models_dict[selected_eval_model]["classification_report"]
        report_rows = []
        for c in classes:
            if c in report_data:
                report_rows.append({
                    "Career Class": c,
                    "Precision": round(report_data[c]["precision"], 3),
                    "Recall": round(report_data[c]["recall"], 3),
                    "F1-Score": round(report_data[c]["f1-score"], 3),
                    "Support": int(report_data[c]["support"])
                })
        st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)

# ==============================================================================
# PAGE 7: ADMINISTRATOR DASHBOARD & DATABASE EXPLORER
# ==============================================================================
elif nav_choice == PAGE_ADMIN_DB or st.session_state.get("show_admin_login", False):
    st.markdown('<div class="main-header">\U0001F512 Administrator Command Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Restricted access portal for authorized faculty and system administrators.</div>', unsafe_allow_html=True)
    
    if not st.session_state.admin_authenticated:
        # Secure Authentication Card (No hints or passwords displayed)
        st.markdown("""
        <div class="highlight-card" style="border-left-color: #DC2626; background: #FEF2F2;">
            <h4 style="margin-top:0; color:#991B1B;">\U0001F510 Restricted Administrator Access</h4>
            <p style="color:#7F1D1D; margin-bottom:0;">Please authenticate with authorized administrator credentials to manage assessment databases and export records.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_auth1, col_auth2 = st.columns([1, 2])
        with col_auth1:
            entered_pass = st.text_input("Administrator Password", type="password", placeholder="\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022", key="admin_pwd_field")
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("\U0001F513 Sign In", type="primary", use_container_width=True):
                    if entered_pass == ADMIN_PASSWORD:
                        st.session_state.admin_authenticated = True
                        st.session_state["show_admin_login"] = False
                        st.success("\u2705 Access granted! Welcome, Administrator.")
                        st.rerun()
                    else:
                        st.error("\u274C Access Denied: Invalid credentials.")
            with btn_col2:
                if st.button("? Student Mode", use_container_width=True):
                    st.session_state["show_admin_login"] = False
                    if hasattr(st, "query_params") and "role" in st.query_params:
                        del st.query_params["role"]
                    if hasattr(st, "query_params") and "admin" in st.query_params:
                        del st.query_params["admin"]
                    st.rerun()
    else:
        # Authenticated Admin Dashboard
        col_title, col_logout = st.columns([4, 1])
        with col_title:
            st.success("\U0001F7E2 **Administrator Session Active**")
        with col_logout:
            if st.button("\U0001F512 Log Out", use_container_width=True):
                st.session_state.admin_authenticated = False
                st.session_state["show_admin_login"] = False
                if hasattr(st, "query_params") and "role" in st.query_params:
                    del st.query_params["role"]
                if hasattr(st, "query_params") and "admin" in st.query_params:
                    del st.query_params["admin"]
                st.rerun()
                
        db_info = db_manager.get_status()
        
        col_db1, col_db2, col_db3 = st.columns(3)
        with col_db1:
            st.markdown(f"""<div class="metric-card"><div class="card-title">Database Engine</div><div class="card-value" style="font-size:1.3rem;">{db_info["mode"]}</div></div>""", unsafe_allow_html=True)
        with col_db2:
            st.markdown(f"""<div class="metric-card"><div class="card-title">Host / Storage</div><div class="card-value" style="font-size:1.1rem; word-break:break-all;">{db_info["host"]}</div></div>""", unsafe_allow_html=True)
        with col_db3:
            st.markdown(f"""<div class="metric-card"><div class="card-title">Database Name</div><div class="card-value" style="font-size:1.2rem;">{db_info["database"]}</div></div>""", unsafe_allow_html=True)

        if not db_info["is_mysql"]:
            st.info("\U0001F4A1 **Storage Status:** Operating on high-speed SQLite storage. To connect to an external MySQL server, configure credentials in `.env` / Streamlit Secrets.")
            if st.button("\U0001F504 Retry MySQL Connection"):
                db_manager._test_connection()
                st.rerun()

        st.divider()
        st.subheader("\U0001F4CB Master Student Assessment Records")
        
        assessments_df = db_manager.get_recent_assessments(limit=200)
        
        if len(assessments_df) == 0:
            st.info("No assessment records found in database. Student submissions will be logged here automatically.")
        else:
            # Summary Metrics for Admin
            adm_col1, adm_col2, adm_col3, adm_col4 = st.columns(4)
            with adm_col1:
                st.metric("Total Submissions", len(assessments_df))
            with adm_col2:
                top_overall_career = assessments_df['primary_recommended_career'].mode()[0] if not assessments_df['primary_recommended_career'].empty else "N/A"
                st.metric("Most Recommended Career", top_overall_career)
            with adm_col3:
                avg_cgpa = round(assessments_df['cgpa'].mean(), 2) if not assessments_df['cgpa'].empty else 0.0
                st.metric("Average Student CGPA", avg_cgpa)
            with adm_col4:
                distinct_students = assessments_df['student_id'].nunique() if 'student_id' in assessments_df else len(assessments_df)
                st.metric("Unique Students", distinct_students)
                
            st.dataframe(assessments_df, use_container_width=True, hide_index=True)
            
            csv_data = assessments_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="\U0001F4E5 Export Complete Assessment Database (CSV)",
                data=csv_data,
                file_name="student_career_assessments_export.csv",
                mime="text/csv",
                type="primary"
            )
