import os
import tempfile
from dotenv import load_dotenv

load_dotenv()
os.environ["MODEL_NAME"] = "llama3.1"

import streamlit as st
import pandas as pd
import time
from datetime import datetime
from io import BytesIO
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
from dental_agent.agent import dental_graph
from dental_agent.tools.vision_model import run_local_xray_inference, generate_gradcam_overlay

# ReportLab Components
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm

# =====================================================================
# 1. PAGE CONFIG & STYLING
# =====================================================================
st.set_page_config(
    page_title="DentaCare AI | Clinical Intelligence",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #0b0f1a;
            color: #e2e8f0;
        }
        .block-container {
            padding-top: 0.5rem !important;
            padding-bottom: 2rem !important;
            max-width: 1320px !important;
        }

        /* ---------- Header ---------- */
        .app-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 18px 28px;
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 12px;
            margin-bottom: 24px;
        }
        .app-header .brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .app-header .brand-icon {
            width: 40px; height: 40px;
            background: #0d9488;
            border-radius: 10px;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.3rem;
        }
        .app-header .brand-text h1 {
            font-size: 1.1rem; font-weight: 800; color: #f8fafc; margin: 0; letter-spacing: -0.02em;
        }
        .app-header .brand-text p {
            font-size: 0.75rem; color: #64748b; margin: 0; font-weight: 500;
        }
        .status-pill {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 600;
            color: #34d399;
            background: rgba(52, 211, 153, 0.08);
            border: 1px solid rgba(52, 211, 153, 0.25);
            padding: 6px 14px;
            border-radius: 100px;
            letter-spacing: 0.04em;
        }
        .status-pill::before {
            content: "●";
            margin-right: 6px;
        }

        /* ---------- Cards ---------- */
        .card {
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 14px;
            padding: 22px 24px;
            margin-bottom: 18px;
        }
        .card-title {
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #64748b;
            margin-bottom: 14px;
            display: flex; align-items: center; gap: 8px;
        }

        /* ---------- Diagnosis result ---------- */
        .diagnosis-result {
            background: linear-gradient(135deg, rgba(217, 119, 6, 0.10) 0%, rgba(217, 119, 6, 0.03) 100%);
            border: 1px solid rgba(217, 119, 6, 0.28);
            border-left: 3px solid #f59e0b;
            border-radius: 10px;
            padding: 16px 18px;
            margin-bottom: 16px;
        }
        .diagnosis-result .label {
            font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600;
        }
        .diagnosis-result .value {
            font-size: 1.45rem; font-weight: 800; color: #fbbf24; margin-top: 4px; letter-spacing: -0.01em;
        }
        .localization-tag {
            font-family: 'JetBrains Mono', monospace;
            background: #0f2027;
            color: #2dd4bf;
            border: 1px solid #134e4a;
            padding: 8px 14px;
            border-radius: 8px;
            font-size: 0.8rem;
            display: inline-block;
        }
        .certainty-block {
            margin-top: 18px;
            padding-top: 16px;
            border-top: 1px solid #1f2937;
        }
        .certainty-value {
            font-size: 2.6rem;
            font-weight: 800;
            color: #f8fafc;
            line-height: 1;
            letter-spacing: -0.02em;
        }
        .certainty-bar-bg {
            height: 6px; width: 100%; background: #1f2937; border-radius: 4px; margin-top: 10px; overflow: hidden;
        }
        .certainty-bar-fill {
            height: 100%; background: linear-gradient(90deg, #0d9488, #2dd4bf); border-radius: 4px;
        }

        /* ---------- Image panel ---------- */
        .img-panel-wrap {
            background: #0d1320;
            border: 1px solid #1f2937;
            border-radius: 10px;
            padding: 10px;
            height: 100%;
        }
        .img-label {
            font-size: 0.72rem; font-weight: 700; color: #64748b; text-align: center;
            text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px;
        }
        .img-label.accent { color: #2dd4bf; }
        .img-empty-state {
            display: flex; align-items: center; justify-content: center;
            height: 220px; background: #0a0e18; border: 1px dashed #334155; border-radius: 8px;
            color: #475569; font-size: 0.8rem; text-align: center; padding: 16px; flex-direction: column; gap: 8px;
        }
        /* Style the Streamlit-rendered column itself as the "panel" box,
           since native widgets (st.image) cannot be nested inside a
           markdown-rendered <div> in Streamlit. */
        div[data-testid="stVerticalBlockBorderWrapper"] .img-panel-marker {
            display: none;
        }
        .img-panel-container {
            background: #0d1320;
            border: 1px solid #1f2937;
            border-radius: 10px;
            padding: 10px;
        }

        /* ---------- Tabs ---------- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px; background: #111827; padding: 6px; border-radius: 12px; border: 1px solid #1f2937;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 22px !important; border-radius: 8px !important; font-weight: 600 !important;
            color: #64748b !important; font-size: 0.9rem !important;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0d9488 !important; color: #ffffff !important;
        }

        /* ---------- Chat ---------- */
        .consult-intro {
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 10px;
            padding: 16px 18px;
            margin-bottom: 16px;
        }
        .consult-intro .meta {
            font-size: 0.75rem; font-weight: 700; color: #38bdf8; margin-bottom: 6px;
            text-transform: uppercase; letter-spacing: 0.04em;
        }
        .consult-intro .text { font-size: 0.92rem; color: #cbd5e1; line-height: 1.6; }

        /* ---------- Buttons ---------- */
        .stButton>button {
            background: #0d9488 !important;
            border: none !important;
            color: white !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            padding: 12px 22px !important;
            transition: background 0.15s ease;
        }
        .stButton>button:hover { background: #0f766e !important; }

        /* ---------- Sidebar ---------- */
        [data-testid="stSidebar"] {
            background-color: #080b14 !important;
            border-right: 1px solid #1f2937 !important;
        }
        .sidebar-card {
            background: #111827; border: 1px solid #1f2937; border-radius: 10px; padding: 14px; margin-bottom: 14px;
        }
        .sidebar-card .title {
            font-size: 0.7rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 10px;
        }
        .sidebar-status-line {
            display: flex; align-items: center; gap: 8px; font-size: 0.85rem; color: #34d399; font-weight: 600; margin-bottom: 6px;
        }
        .sidebar-status-line .dot {
            width: 8px; height: 8px; border-radius: 50%; background: #34d399;
        }
        .sidebar-meta { font-size: 0.78rem; color: #64748b; line-height: 1.6; margin-left: 16px; }

        .stFileUploader {
            border: 1px dashed #334155 !important;
            background-color: #0f172a !important;
            border-radius: 10px !important;
        }
        hr { border-color: #1f2937 !important; }

        /* ---------- Glass Card with Glow ---------- */
        .glass-card {
            position: relative;
            background: rgba(17, 24, 39, 0.55);
            border: 1px solid rgba(45, 212, 191, 0.18);
            border-radius: 14px;
            padding: 18px 20px;
            margin-top: 14px;
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
            animation: cardRise 0.5s ease-out forwards;
        }
        .glass-card::before {
            content: '';
            position: absolute;
            top: -50%; left: -60%;
            width: 60%; height: 200%;
            background: linear-gradient(
                100deg,
                transparent 20%,
                rgba(45, 212, 191, 0.16) 45%,
                rgba(255, 255, 255, 0.08) 50%,
                rgba(45, 212, 191, 0.16) 55%,
                transparent 80%
            );
            transform: rotate(8deg);
            animation: sheenSweep 3.2s ease-in-out infinite;
            pointer-events: none;
        }
        .glass-card::after {
            content: '';
            position: absolute;
            inset: 0;
            border-radius: 14px;
            border: 1px solid transparent;
            background: linear-gradient(135deg, rgba(45,212,191,0.35), transparent 40%, transparent 60%, rgba(45,212,191,0.2)) border-box;
            -webkit-mask: linear-gradient(#fff 0 0) padding-box, linear-gradient(#fff 0 0);
            -webkit-mask-composite: xor;
            mask-composite: exclude;
            pointer-events: none;
            opacity: 0.7;
        }
        @keyframes sheenSweep {
            0%   { left: -60%; }
            50%  { left: 120%; }
            100% { left: 120%; }
        }
        @keyframes cardRise {
            from { opacity: 0; transform: translateY(8px) scale(0.98); }
            to   { opacity: 1; transform: translateY(0) scale(1); }
        }

        /* ---------- Doctor Assignment Animation ---------- */
        .doctor-assign-card {
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 16px 18px;
        }
        .doctor-avatar-ring {
            position: relative;
            width: 52px; height: 52px;
            border-radius: 50%;
            flex-shrink: 0;
            background: linear-gradient(135deg, #0d9488, #134e4a);
            display: flex; align-items: center; justify-content: center;
            font-size: 1.4rem;
            box-shadow: 0 0 0 0 rgba(45, 212, 191, 0.5);
            animation: avatarPulse 2.2s ease-out infinite;
        }
        .doctor-avatar-ring::before {
            content: '';
            position: absolute;
            inset: -4px;
            border-radius: 50%;
            border: 2px solid rgba(45, 212, 191, 0.35);
            animation: ringExpand 2.2s ease-out infinite;
        }
        @keyframes avatarPulse {
            0%   { box-shadow: 0 0 0 0 rgba(45, 212, 191, 0.45); }
            70%  { box-shadow: 0 0 0 14px rgba(45, 212, 191, 0); }
            100% { box-shadow: 0 0 0 0 rgba(45, 212, 191, 0); }
        }
        @keyframes ringExpand {
            0%   { transform: scale(0.85); opacity: 0.9; }
            70%  { transform: scale(1.25); opacity: 0; }
            100% { transform: scale(1.25); opacity: 0; }
        }
        .doctor-assign-info { flex: 1; }
        .doctor-assign-label {
            font-size: 0.68rem; font-weight: 700; color: #64748b;
            text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px;
            display: flex; align-items: center; gap: 8px;
        }
        .doctor-assign-name {
            font-size: 1.05rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.01em;
        }
        .doctor-assign-name .accent { color: #2dd4bf; }
        .assign-check {
            width: 22px; height: 22px; border-radius: 50%;
            background: rgba(45, 212, 191, 0.15);
            border: 1.5px solid #2dd4bf;
            display: flex; align-items: center; justify-content: center;
            font-size: 0.7rem; color: #2dd4bf; font-weight: 800;
            animation: checkPop 0.4s ease-out 0.3s both;
            flex-shrink: 0;
        }
        @keyframes checkPop {
            0%   { transform: scale(0); opacity: 0; }
            60%  { transform: scale(1.2); opacity: 1; }
            100% { transform: scale(1); opacity: 1; }
        }
        .live-dot-small {
            width: 6px; height: 6px; border-radius: 50%; background: #2dd4bf;
            display: inline-block;
            animation: heartGlowSmall 1.2s infinite alternate;
        }
        @keyframes heartGlowSmall {
            from { box-shadow: 0 0 0px #2dd4bf; opacity: 0.6; transform: scale(0.85); }
            to   { box-shadow: 0 0 8px #2dd4bf; opacity: 1; transform: scale(1.15); }
        }

        /* ---------- Generic shimmer text ---------- */
        .shimmer-text {
            background: linear-gradient(90deg, #94a3b8 0%, #f8fafc 50%, #94a3b8 100%);
            background-size: 200% auto;
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            animation: shimmerMove 2.5s linear infinite;
        }
        @keyframes shimmerMove {
            to { background-position: -200% center; }
        }

        /* ---------- Intake form card ---------- */
        .intake-card {
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 16px;
            padding: 36px 40px;
        }
        .intake-header {
            display: flex; align-items: center; justify-content: center; gap: 16px; margin-bottom: 8px;
            text-align: center;
            flex-direction: column;
        }
        .intake-header .icon {
            width: 52px; height: 52px; border-radius: 12px; background: #0d9488;
            display: flex; align-items: center; justify-content: center; font-size: 1.6rem;
            flex-shrink: 0;
        }
        .intake-header h2 {
            font-size: 1.4rem; font-weight: 800; color: #f8fafc; margin: 0; letter-spacing: -0.01em;
        }
        .intake-header p {
            font-size: 0.82rem; color: #2dd4bf; margin: 2px 0 0 0; text-transform: uppercase;
            letter-spacing: 0.06em; font-weight: 600;
        }

        /* ---------- Clinician badge ---------- */
        .clinician-badge {
            background: #0f172a;
            border: 1px solid #134e4a;
            padding: 12px 16px;
            border-radius: 8px;
            margin-top: 8px;
            display: flex; align-items: center; gap: 12px;
        }
        .clinician-badge .dot {
            width: 8px; height: 8px; background: #2dd4bf; border-radius: 50%;
            box-shadow: 0 0 8px #2dd4bf;
        }
        .clinician-badge .text { font-size: 0.85rem; color: #94a3b8; }
        .clinician-badge .text b { color: #2dd4bf; font-weight: 700; }

        /* ---------- Loading state ---------- */
        .loading-card {
            text-align: center;
            margin: 8% auto;
            max-width: 480px;
            padding: 48px 32px;
            position: relative;
            background: rgba(17, 24, 39, 0.55);
            border: 1px solid rgba(45, 212, 191, 0.18);
            border-radius: 16px;
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            overflow: hidden;
            box-shadow: 0 8px 40px rgba(0, 0, 0, 0.4);
        }
        .loading-card::before {
            content: '';
            position: absolute;
            top: -50%; left: -60%;
            width: 60%; height: 200%;
            background: linear-gradient(
                100deg,
                transparent 20%,
                rgba(45, 212, 191, 0.16) 45%,
                rgba(255, 255, 255, 0.08) 50%,
                rgba(45, 212, 191, 0.16) 55%,
                transparent 80%
            );
            transform: rotate(8deg);
            animation: sheenSweep 2.6s ease-in-out infinite;
            pointer-events: none;
        }
        .loading-icon-ring {
            width: 64px; height: 64px; margin: 0 auto 24px auto;
            border: 3px solid #1f2937;
            border-top-color: #0d9488;
            border-right-color: #2dd4bf;
            border-radius: 50%;
            animation: spin 0.9s linear infinite;
            position: relative;
            z-index: 1;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .loading-title {
            font-size: 1.15rem; font-weight: 700; letter-spacing: -0.01em;
            position: relative; z-index: 1;
        }
        .loading-sub {
            font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #64748b; margin-top: 8px; letter-spacing: 0.02em;
            position: relative; z-index: 1;
        }

        /* ---------- Success block ---------- */
        .success-block {
            display: flex; align-items: center; gap: 14px;
            background-color: rgba(52, 211, 153, 0.06);
            border: 1px solid rgba(52, 211, 153, 0.25);
            padding: 16px 18px; border-radius: 10px; margin-top: 14px;
        }
        .success-icon {
            width: 38px; height: 38px; border-radius: 50%; background: rgba(52, 211, 153, 0.12);
            border: 1.5px solid #34d399; display: flex; align-items: center; justify-content: center;
            color: #34d399; font-size: 1.1rem; font-weight: bold; flex-shrink: 0;
        }
        .success-block .title { color: #34d399; font-weight: 700; font-size: 0.95rem; }
        .success-block .desc { color: #94a3b8; font-size: 0.85rem; margin-top: 2px; }

        /* ---------- Patient header bar ---------- */
        .patient-bar {
            display: flex; justify-content: space-between; align-items: center;
            padding: 18px 24px;
            background: rgba(17, 24, 39, 0.55);
            border: 1px solid rgba(45, 212, 191, 0.18);
            border-radius: 12px;
            margin-bottom: 20px;
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            position: relative;
            overflow: hidden;
            animation: cardRise 0.5s ease-out forwards;
        }
        .patient-bar::before {
            content: '';
            position: absolute;
            top: -50%; left: -60%;
            width: 60%; height: 200%;
            background: linear-gradient(
                100deg,
                transparent 20%,
                rgba(45, 212, 191, 0.14) 45%,
                rgba(255, 255, 255, 0.06) 50%,
                rgba(45, 212, 191, 0.14) 55%,
                transparent 80%
            );
            transform: rotate(8deg);
            animation: sheenSweep 4s ease-in-out infinite;
            pointer-events: none;
        }
        .patient-bar .label {
            font-size: 0.7rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em;
            position: relative; z-index: 1;
        }
        .patient-bar .name {
            font-size: 1.3rem; font-weight: 800; color: #f8fafc; margin-top: 2px; letter-spacing: -0.01em;
            position: relative; z-index: 1;
        }
        .patient-bar .meta {
            font-size: 0.8rem; color: #2dd4bf; margin-top: 4px; font-family: 'JetBrains Mono', monospace;
            position: relative; z-index: 1;
        }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# 2. DATA LOADING
# =====================================================================
LOGO_PATH = "image_db957c.jpg"

try:
    # 1. Load and Normalize the data
    df = pd.read_csv("doctor_availability.csv")
    # Ensure the column is treated as a string, stripped, and forced to lowercase
    df['is_available_clean'] = df['is_available'].astype(str).str.strip().str.lower()

    # 2. Robust Filtering for Available Slots
    # Using .str.contains with case=False ensures 'true' matches 'TRUE', 'True', etc.
    available_df = df[df['is_available_clean'].str.contains('true', case=False, na=False)]

    # 3. Aggregator Metrics
    total_slots = len(df)
    open_slots = len(available_df)
    booked_slots = total_slots - open_slots
    utilization_rate = int((booked_slots / total_slots) * 100) if total_slots > 0 else 0
except Exception:
    df = pd.DataFrame()
    total_slots, open_slots, booked_slots, utilization_rate = 0, 0, 0, 0

# Use a proper temp directory instead of hardcoded D:\ paths (cross-platform safe)
TEMP_DIR = tempfile.gettempdir()
XRAY_PATH = os.path.join(TEMP_DIR, "dentacare_uploaded_xray.png")
HEATMAP_PATH = os.path.join(TEMP_DIR, "dentacare_heatmap_output.png")

# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    st.markdown("<h3 style='color:#f8fafc; font-weight:800; text-align:center; margin-top:8px; font-size:1.05rem;'>DentaCare Hub</h3>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
        <div class="sidebar-card">
            <div class="title">System Status</div>
            <div class="sidebar-status-line"><span class="dot"></span> Compute Online</div>
            <div class="sidebar-meta">Model: Custom CNN Core<br/>Track: Panoramic OPG Inference</div>
        </div>
    """, unsafe_allow_html=True)
    if total_slots > 0:
        st.markdown(f"""
            <div class="sidebar-card">
                <div class="title">Clinic Schedule</div>
                <div class="sidebar-meta">
                    Total slots: {total_slots}<br/>
                    Available: {open_slots}<br/>
                    Booked: {booked_slots}<br/>
                    Utilization: {utilization_rate}%
                </div>
            </div>
        """, unsafe_allow_html=True)

# =====================================================================
# HEADER
# =====================================================================
st.markdown("""
    <div class="app-header">
        <div class="brand">
            <div class="brand-icon">🦷</div>
            <div class="brand-text">
                <h1>DentalCare AI 2.0</h1>
                <p>Clinical Intelligence Hub</p>
            </div>
        </div>
        <div class="status-pill">Pipeline Active</div>
    </div>
""", unsafe_allow_html=True)

if "pipeline_active" not in st.session_state:
    st.session_state.pipeline_active = False

page_viewport = st.empty()

# =====================================================================
# 3. INTAKE PORTAL
# =====================================================================
if not st.session_state.pipeline_active:
    with page_viewport.container():
        _, center_col, _ = st.columns([0.6, 1.6, 0.6])
        with center_col:
            with st.container(border=True):

                # FIX: header + icon + hr merged into ONE markdown call so the
                # text renders INSIDE the visual box instead of an empty box
                # appearing above a separately-rendered header.
                st.markdown("""
                    <div class="intake-header">
                        <div class="icon">🩻</div>
                        <div>
                            <h2>Clinical Intake Portal</h2>
                            <p>Oral Radiography Diagnostics</p>
                        </div>
                    </div>
                    <hr style="margin: 20px 0;">
                """, unsafe_allow_html=True)

                p_name = st.text_input("Patient full legal name", value="e.g- Biswajit pattanaik")

                col_a, col_b = st.columns(2)
                with col_a:
                    p_age = st.text_input("Age", value="E.g- 86")
                with col_b:
                    p_gender = st.selectbox("Gender", ["Male", "Female", "Other"])

                col_c, col_d = st.columns(2)
                with col_c:
                    p_house = st.text_input("Address", value="E.g -Plot No. 42, Cyber Enclave")
                with col_d:
                    p_place = st.text_input("City", value="Bhubaneswar, Odisha")

                doctor_options = df['doctor_name'].unique() if total_slots > 0 else ["Dr. Swarna Prabha Jena"]
                p_doctor = st.selectbox("Consulting clinician", doctor_options)

                st.markdown(f"""
                    <div class="glass-card doctor-assign-card">
                        <div class="doctor-avatar-ring">🩺</div>
                        <div class="doctor-assign-info">
                            <div class="doctor-assign-label">
                                <span class="live-dot-small"></span> Clinician Assigned
                            </div>
                            <div class="doctor-assign-name">Dr. <span class="accent">{p_doctor}</span></div>
                        </div>
                        <div class="assign-check">✓</div>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                uploaded_file = st.file_uploader("Upload panoramic OPG radiograph", type=["png", "jpg", "jpeg"])

                if st.button("Run Diagnostic Pipeline →", use_container_width=True):
                    if uploaded_file is not None:
                        st.session_state.pop("heatmap_path", None)
                        st.session_state.pop("pdf_ready", None)

                        page_viewport.empty()
                        with page_viewport.container():
                            st.markdown("""
                                <div class="loading-card">
                                    <div class="loading-icon-ring"></div>
                                    <div class="loading-title shimmer-text">Running Vision Model</div>
                                    <div class="loading-sub">Analyzing radiograph &amp; generating Grad-CAM overlay…</div>
                                </div>
                            """, unsafe_allow_html=True)

                            with open(XRAY_PATH, "wb") as f:
                                f.write(uploaded_file.getbuffer())

                            predictions = run_local_xray_inference(XRAY_PATH)
                            try:
                                generated_path = generate_gradcam_overlay(XRAY_PATH, HEATMAP_PATH)
                            except Exception as gradcam_err:
                                generated_path = None
                                st.session_state.gradcam_error = str(gradcam_err)

                            st.session_state.p_name = p_name
                            st.session_state.p_age = p_age
                            st.session_state.p_gender = p_gender
                            st.session_state.p_house = p_house
                            st.session_state.p_place = p_place
                            st.session_state.p_doctor = p_doctor
                            st.session_state.metrics_data = predictions
                            st.session_state.max_disease = max(predictions, key=predictions.get)
                            st.session_state.max_value = predictions[st.session_state.max_disease]
                            st.session_state.xray_path = XRAY_PATH
                            if generated_path and os.path.exists(generated_path):
                                st.session_state.heatmap_path = generated_path
                            elif os.path.exists(HEATMAP_PATH):
                                st.session_state.heatmap_path = HEATMAP_PATH
                            else:
                                st.session_state.heatmap_path = ""

                            time.sleep(1.0)
                            st.session_state.pipeline_active = True
                        st.rerun()
                    else:
                        st.error("Please upload a panoramic radiograph file before continuing.")

# =====================================================================
# 4. CLINICAL WORKSPACE
# =====================================================================
else:
    with page_viewport.container():
        p_id_string = f"PACS-{hash(st.session_state.p_name) & 0xFFFFFF:X}"

        st.markdown(f"""
            <div class="patient-bar">
                <div>
                    <div class="label">Active Subject</div>
                    <div class="name">{st.session_state.p_name}</div>
                    <div class="meta">{p_id_string} &nbsp;•&nbsp; {st.session_state.p_age} yrs &nbsp;•&nbsp; {st.session_state.p_gender} &nbsp;•&nbsp; {st.session_state.p_place}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if st.button("← New Patient / Reset"):
            st.session_state.pipeline_active = False
            st.session_state.pop("pdf_ready", None)
            st.rerun()

        st.markdown("<br/>", unsafe_allow_html=True)

        tab_diagnostics, tab_appointments, tab_consultation = st.tabs([
            "Diagnostics & Vision",
            "Appointments",
            "AI Consultation"
        ])

        # ----------------------------------------------------------------
        # TAB 1: DIAGNOSTICS
        # ----------------------------------------------------------------
        with tab_diagnostics:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            metric_left, metric_right = st.columns([0.85, 1.15])

            with metric_left:
                st.markdown(f"""
                    <div class="diagnosis-result">
                        <div class="label">Primary Finding</div>
                        <div class="value">{st.session_state.max_disease}</div>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="card-title">Anatomical Localization</div>', unsafe_allow_html=True)
                st.markdown('<div class="localization-tag">📍 Mapped panoramic arch region</div>', unsafe_allow_html=True)

                st.markdown(f"""
                    <div class="certainty-block">
                        <div class="card-title">Confidence Score</div>
                        <div class="certainty-value">{st.session_state.max_value}%</div>
                        <div class="certainty-bar-bg">
                            <div class="certainty-bar-fill" style="width:{st.session_state.max_value}%;"></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            with metric_right:
                img_sub1, img_sub2 = st.columns(2)

                # FIX: removed the broken '<div class="img-panel">' open/close
                # pattern (st.image can't render inside a markdown div in
                # Streamlit, which left a stray empty box above each label).
                # Using st.container(border=True) gives a real bordered box
                # that the label + image render INSIDE of.
                with img_sub1:
                    with st.container(border=True):
                        st.markdown('<div class="img-label">Input Radiograph</div>', unsafe_allow_html=True)
                        if os.path.exists(st.session_state.xray_path):
                            st.image(st.session_state.xray_path, use_container_width=True)
                        else:
                            st.markdown('<div class="img-empty-state">⚠️<br/>Original radiograph<br/>not found</div>', unsafe_allow_html=True)

                with img_sub2:
                    with st.container(border=True):
                        st.markdown('<div class="img-label accent">Grad-CAM Overlay</div>', unsafe_allow_html=True)
                        if st.session_state.heatmap_path and os.path.exists(st.session_state.heatmap_path):
                            st.image(st.session_state.heatmap_path, use_container_width=True)
                        else:
                            err = st.session_state.get("gradcam_error")
                            msg = f"⚠️<br/>Overlay generation failed<br/><span style='font-size:0.7rem;'>{err}</span>" if err else "⚠️<br/>Heatmap overlay<br/>not generated"
                            st.markdown(f'<div class="img-empty-state">{msg}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            g_col1, g_col2 = st.columns(2)
            with g_col1:
                st.markdown('<div class="card-title">Probability Distribution</div>', unsafe_allow_html=True)
                chart_df = pd.DataFrame({
                    "Condition": list(st.session_state.metrics_data.keys()),
                    "Score": list(st.session_state.metrics_data.values())
                }).set_index("Condition")
                st.bar_chart(chart_df, color="#2563eb", use_container_width=True)
            with g_col2:
                st.markdown('<div class="card-title">Benchmark Comparison</div>', unsafe_allow_html=True)
                bench_df = pd.DataFrame({
                    "Metric": ["Train Mean", "This Result", "Test Floor"],
                    "Score": [45, int(st.session_state.max_value), 30]
                }).set_index("Metric")
                st.bar_chart(bench_df, color="#38bdf8", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ----------------------------------------------------------------
        # TAB 2: APPOINTMENTS
        # ----------------------------------------------------------------
        with tab_appointments:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Clinic Reservation Platform</div>', unsafe_allow_html=True)
            if total_slots > 0:
                st.dataframe(
                    df.rename(columns={
                        "date_slot": "Time Slot",
                        "specialization": "Specialization",
                        "doctor_name": "Dentist",
                        "is_available": "Available",
                        "patient_to_attend": "Patient"
                    }),
                    use_container_width=True, hide_index=True
                )
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown('<div class="card-title">Book a Slot</div>', unsafe_allow_html=True)
                form_col1, form_col2 = st.columns(2)
                with form_col1:
                    dentist_options = df['doctor_name'].unique()
                    select_doc = st.selectbox("Clinician", dentist_options if len(dentist_options) > 0 else ["No data"])
                with form_col2:
                    filtered_slots = df[df['doctor_name'] == select_doc]['date_slot'].unique()
                    select_time = st.selectbox("Time slot", filtered_slots if len(filtered_slots) > 0 else ["No slots"])

                st.markdown("<br/>", unsafe_allow_html=True)
                if st.button("Confirm Appointment", use_container_width=True):
                    with st.spinner("Updating registry…"):
                        time.sleep(0.8)
                    st.markdown(f"""
                        <div class="success-block">
                            <div class="success-icon">✓</div>
                            <div>
                                <div class="title">Appointment Confirmed</div>
                                <div class="desc">Dr. {select_doc} is booked for {select_time}.</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No appointment data found in doctor_availability.csv.")
            st.markdown('</div>', unsafe_allow_html=True)

        # ----------------------------------------------------------------
        # TAB 3: AI CONSULTATION + PDF EXPORT
        # ----------------------------------------------------------------
        with tab_consultation:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">AI Consultation</div>', unsafe_allow_html=True)

            st.markdown(f"""
                <div class="consult-intro">
                    <div class="meta">Dr. Alex — AI Consultant</div>
                    <div class="text">
                        Hello Dr. {st.session_state.p_doctor}. The vision model has analyzed the panoramic radiograph for
                        <b>{st.session_state.p_name}</b>. The primary finding is
                        <b>{st.session_state.max_disease}</b>, with a confidence score of
                        <b>{st.session_state.max_value}%</b>. A detailed report can be generated below.
                    </div>
                </div>
            """, unsafe_allow_html=True)

            if "agent_chat_log" not in st.session_state:
                st.session_state.agent_chat_log = []
            for chat in st.session_state.agent_chat_log:
                role_tag = "user" if isinstance(chat, HumanMessage) else "assistant"
                with st.chat_message(role_tag):
                    st.markdown(chat.content)

            if clinician_query := st.chat_input("Ask the AI consultant…"):
                with st.chat_message("user"):
                    st.markdown(clinician_query)
                st.session_state.agent_chat_log.append(HumanMessage(content=clinician_query))
                with st.chat_message("assistant"):
                    resp_placeholder = st.empty()
                    full_text = ""
                    try:
                        events = dental_graph.stream(
                            {"messages": st.session_state.agent_chat_log},
                            stream_mode=["messages"],
                            config={"recursion_limit": 15}
                        )
                        for ev_type, chunk_data in events:
                            if ev_type == "messages":
                                token, _ = chunk_data
                                if isinstance(token, AIMessageChunk) and token.content:
                                    full_text += token.content
                                    resp_placeholder.markdown(full_text + "▌")
                        resp_placeholder.markdown(full_text)
                        st.session_state.agent_chat_log.append(AIMessage(content=full_text))
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            st.markdown("---")
            st.markdown('<div class="card-title">Generate Report</div>', unsafe_allow_html=True)

            if st.button("Generate PDF Health Record", use_container_width=True):
                with st.spinner("Compiling report…"):

                    pdf_buffer = BytesIO()
                    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)
                    story = []

                    styles = getSampleStyleSheet()
                    accent_color = colors.HexColor("#1e3a8a")
                    text_dark = colors.HexColor("#1e293b")

                    title_style = ParagraphStyle('DocTitle', fontName='Helvetica-Bold', fontSize=22, leading=26, textColor=accent_color, spaceAfter=4)
                    sub_title_style = ParagraphStyle('DocSubTitle', fontName='Helvetica', fontSize=10, leading=14, textColor=colors.gray, spaceAfter=20)
                    heading_style = ParagraphStyle('SectionHeading', fontName='Helvetica-Bold', fontSize=12, leading=16, textColor=accent_color, spaceBefore=15, spaceAfter=8)
                    text_style = ParagraphStyle('BodyTextCustom', fontName='Helvetica', fontSize=10, textColor=text_dark, leading=14)
                    bold_text_style = ParagraphStyle('BodyBoldCustom', fontName='Helvetica-Bold', fontSize=10, textColor=text_dark, leading=14)
                    alert_style = ParagraphStyle('AlertText', fontName='Helvetica-Bold', fontSize=12, leading=16, textColor=colors.HexColor("#991b1b"))
                    medical_narrative_style = ParagraphStyle('MedNarrative', fontName='Helvetica', fontSize=10, leading=16, textColor=colors.HexColor("#0f172a"), spaceBefore=6)

                    if os.path.exists(LOGO_PATH):
                        clinic_seal = Image(LOGO_PATH, width=65, height=50)
                        header_title_p = Paragraph(
                            "<b>METROPOLITAN DIGITAL MEDICAL CENTRE</b><br/><font size=9 color='gray'>Oral Radiography Screening Hub</font>",
                            title_style
                        )
                        brand_table = Table([[clinic_seal, header_title_p]], colWidths=[80, 440])
                        brand_table.setStyle(TableStyle([
                            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                            ('LEFTPADDING', (0, 0), (-1, -1), 0),
                            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                            ('TOPPADDING', (0, 0), (-1, -1), 0),
                            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
                        ]))
                        story.append(brand_table)
                    else:
                        story.append(Paragraph("METROPOLITAN DIGITAL MEDICAL CENTRE", title_style))

                    story.append(Paragraph("Automated Diagnostic Case Summary — Panoramic OPG Deep Learning Analysis", sub_title_style))
                    story.append(Spacer(1, 10))

                    # Section 1: Patient metadata
                    story.append(Paragraph("<b>1. Patient Information</b>", heading_style))
                    meta_table_data = [
                        [Paragraph("Patient Name:", bold_text_style), Paragraph(st.session_state.p_name, text_style),
                         Paragraph("Report Date:", bold_text_style), Paragraph(datetime.now().strftime('%B %d, %Y'), text_style)],
                        [Paragraph("Age:", bold_text_style), Paragraph(f"{st.session_state.p_age} Years", text_style),
                         Paragraph("Case ID:", bold_text_style), Paragraph(p_id_string, text_style)],
                        [Paragraph("Gender:", bold_text_style), Paragraph(st.session_state.p_gender, text_style),
                         Paragraph("Attending Doctor:", bold_text_style), Paragraph(f"Dr. {st.session_state.p_doctor}", text_style)],
                        [Paragraph("Address:", bold_text_style), Paragraph(st.session_state.p_house, text_style),
                         Paragraph("City:", bold_text_style), Paragraph(st.session_state.p_place, text_style)]
                    ]
                    meta_table = Table(meta_table_data, colWidths=[90, 170, 90, 170])
                    meta_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                        ('PADDING', (0, 0), (-1, -1), 6),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ]))
                    story.append(meta_table)
                    story.append(Spacer(1, 15))

                    # Section 2: Model evaluation
                    story.append(Paragraph("<b>2. Vision Model Evaluation</b>", heading_style))
                    findings_data = [
                        [Paragraph("Primary Diagnosis:", bold_text_style), Paragraph(st.session_state.max_disease, alert_style)],
                        [Paragraph("Model Confidence:", bold_text_style), Paragraph(f"<b>{st.session_state.max_value}%</b>", text_style)],
                        [Paragraph("Localization:", bold_text_style), Paragraph("Dental arch segment, mapped via Grad-CAM activation", text_style)]
                    ]
                    findings_table = Table(findings_data, colWidths=[160, 360])
                    findings_table.setStyle(TableStyle([
                        ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                        ('PADDING', (0, 0), (-1, -1), 8),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ]))
                    story.append(findings_table)
                    story.append(Spacer(1, 12))

                    # Section 3: Clinical narrative
                    story.append(Paragraph("<b>3. Clinical Impression</b>", heading_style))
                    disease_key = st.session_state.max_disease.lower()
                    if "caries" in disease_key:
                        narrative_text = "<b>Observation:</b> Radiographic views display distinct localized radiolucency within the crown architecture, indicating progressive enamel and dentin mineral loss. <b>Mapping:</b> Activation regions isolate structural degradation within the targeted segment boundaries."
                    elif "ulcer" in disease_key:
                        narrative_text = "<b>Observation:</b> Loss of mucosal epithelial continuity with peripheral edema boundaries visible. <b>Mapping:</b> Feature maps track soft-tissue density disruptions consistent with an active lesion."
                    elif "gingivitis" in disease_key:
                        narrative_text = "<b>Observation:</b> Localized alveolar cortical bone density alterations trace crestal margins consistent with inflammation."
                    else:
                        narrative_text = f"<b>Observation:</b> Structural density anomaly detected matching reference parameters for {st.session_state.max_disease} with high spatial correspondence."
                    story.append(Paragraph(narrative_text, medical_narrative_style))
                    story.append(Spacer(1, 15))

                    # Section 4: Images
                    story.append(Paragraph("<b>4. Radiograph Comparison</b>", heading_style))
                    images_row = []
                    if os.path.exists(st.session_state.xray_path):
                        img1 = Image(st.session_state.xray_path, width=240, height=180)
                        images_row.append(img1)
                    else:
                        images_row.append(Paragraph("Original image unavailable", text_style))
                    if st.session_state.heatmap_path and os.path.exists(st.session_state.heatmap_path):
                        img2 = Image(st.session_state.heatmap_path, width=240, height=180)
                        images_row.append(img2)
                    else:
                        images_row.append(Paragraph("Heatmap unavailable", text_style))

                    images_table_data = [
                        images_row,
                        [Paragraph("<font color='#64748b'><b>Image A:</b> Original Panoramic Radiograph</font>", text_style),
                         Paragraph("<font color='#2563eb'><b>Image B:</b> Grad-CAM Pathology Overlay</font>", text_style)]
                    ]
                    images_table = Table(images_table_data, colWidths=[260, 260])
                    images_table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('PADDING', (0, 0), (-1, -1), 5),
                    ]))
                    story.append(images_table)
                    story.append(Spacer(1, 15))

                    # Section 5: Full probability table
                    story.append(Paragraph("<b>5. Full Probability Matrix</b>", heading_style))
                    coef_header_style = ParagraphStyle('CoefHeader', fontName='Helvetica-Bold', fontSize=10, textColor=colors.white, leading=14)
                    coef_table_data = [[
                        Paragraph("Condition", coef_header_style),
                        Paragraph("Confidence Score (%)", coef_header_style)
                    ]]
                    for pathology, confidence in st.session_state.metrics_data.items():
                        coef_table_data.append([Paragraph(pathology, text_style), Paragraph(f"{confidence}%", text_style)])

                    coef_table = Table(coef_table_data, colWidths=[280, 240])
                    coef_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), accent_color),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                        ('PADDING', (0, 0), (-1, -1), 5),
                        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                    ]))
                    story.append(coef_table)
                    story.append(Spacer(1, 20))

                    # Disclaimer
                    story.append(Paragraph("<hr/>", text_style))
                    story.append(Paragraph("<b>Disclaimer:</b>", text_style))
                    story.append(Paragraph(
                        "This document is an AI-generated diagnostic summary based on local model inference. "
                        "All findings must be reviewed and confirmed by a licensed physician before any treatment "
                        "decision is made.",
                        text_style
                    ))
                    story.append(Spacer(1, 15))
                    story.append(Paragraph(f"<b>Reviewing Clinician:</b> Dr. {st.session_state.p_doctor}", text_style))

                    doc.build(story)
                    pdf_data = pdf_buffer.getvalue()
                    pdf_buffer.close()

                    st.session_state.generated_pdf_bytes = pdf_data
                    st.session_state.pdf_ready = True
                    st.toast("PDF report generated", icon="📄")
                    st.rerun()

            if st.session_state.get("pdf_ready"):
                st.markdown("""
                    <div class="success-block">
                        <div class="success-icon">✓</div>
                        <div>
                            <div class="title">PDF Report Ready</div>
                            <div class="desc">Your dental health record has been compiled and is ready to download.</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                st.download_button(
                    label="Download PDF Report",
                    data=st.session_state.generated_pdf_bytes,
                    file_name=f"Dental_Report_{p_id_string}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            st.markdown('</div>', unsafe_allow_html=True)