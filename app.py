import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import pickle
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DermaAI - Pakistan Skin Cancer Referral System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN UI STYLING & CUSTOM CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f9;
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .card {
        background: white;
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }
    .metric-box {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(30, 60, 114, 0.3);
        margin-bottom: 15px;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 6px solid #dc3545;
        padding: 18px;
        border-radius: 6px;
        color: #721c24;
        font-size: 15px;
    }
    .safe-box {
        background-color: #d4edda;
        border-left: 6px solid #28a745;
        padding: 18px;
        border-radius: 6px;
        color: #155724;
        font-size: 15px;
    }
    .doctor-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #1e3c72;
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    h1, h2, h3 {
        color: #1a252f;
        font-family: 'Helvetica Neue', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# --- PAKISTAN NATIONWIDE HOSPITALS & SPECIALISTS DATABASE ---
PAKISTAN_SPECIALISTS = {
    "Lahore": [
        {"hospital": "Shaukat Khanum Memorial Cancer Hospital", "location": "Johar Town, Lahore", "specialist": "Dr. Aasim Yusuf", "role": "Chief Medical Officer & Oncology Expert", "rating": "⭐ 4.9 / 5.0", "contact": "042-35905000"},
        {"hospital": "Doctors Hospital & Medical Centre", "location": "Canal Bank, Lahore", "specialist": "Dr. Azam Samdani", "role": "Consultant Dermatologist & Skin Surgeon", "rating": "⭐ 4.8 / 5.0", "contact": "042-111-223-311"},
        {"hospital": "Hamid Latif Hospital", "location": "Gulberg III, Lahore", "specialist": "Prof. Dr. Tahir Saeed Haroon", "role": "Senior Dermatologist", "rating": "⭐ 4.7 / 5.0", "contact": "042-35750817"}
    ],
    "Karachi": [
        {"hospital": "Aga Khan University Hospital (AKUH)", "location": "Stadium Road, Karachi", "specialist": "Dr. Zohra Zaidi", "role": "Consultant Dermatologist", "rating": "⭐ 4.9 / 5.0", "contact": "021-111-911-911"},
        {"hospital": "Indus Hospital & Health Network", "location": "Korangi, Karachi", "specialist": "Oncology Unit", "role": "Dermatological Surgery & Screening", "rating": "⭐ 4.8 / 5.0", "contact": "021-35112709"},
        {"hospital": "Ziauddin Hospital", "location": "Clifton, Karachi", "specialist": "Dr. Shahid Hassan", "role": "Senior Cancer Treatment Specialist", "rating": "⭐ 4.7 / 5.0", "contact": "021-35862937"}
    ],
    "Islamabad / Rawalpindi": [
        {"hospital": "Shifa International Hospital", "location": "H-8/4, Islamabad", "specialist": "Dr. Humayun Mahmud", "role": "Consultant Dermatologist & Laser Specialist", "rating": "⭐ 4.9 / 5.0", "contact": "051-8464646"},
        {"hospital": "NORI Hospital", "location": "H-8/4, Islamabad", "specialist": "Dr. Muhammad Faheem", "role": "Head of Oncology Department", "rating": "⭐ 4.8 / 5.0", "contact": "051-9250553"},
        {"hospital": "Armed Forces Institute of Pathology (AFIP)", "location": "Rawalpindi", "specialist": "Histopathology Unit", "role": "Biopsy & Malignancy Confirmation", "rating": "⭐ 4.9 / 5.0", "contact": "051-9271026"}
    ],
    "Peshawar": [
        {"hospital": "Northwest General Hospital", "location": "Hayatabad, Peshawar", "specialist": "Dr. Jawad Ahmad", "role": "Consultant Oncologist", "rating": "⭐ 4.8 / 5.0", "contact": "091-5838888"},
        {"hospital": "Hayatabad Medical Complex (HMC)", "location": "Hayatabad, Peshawar", "specialist": "Dermatology Department", "role": "Public Healthcare Specialist", "rating": "⭐ 4.6 / 5.0", "contact": "091-9217140"}
    ]
}

# --- LOAD MODEL, CLASSES & TRAINING HISTORY ---
@st.cache_resource
def load_backend_system():
    model = load_model('skin_cancer_model.h5')
    with open('class_names.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    history = None
    if os.path.exists('history_dict.pkl'):
        with open('history_dict.pkl', 'rb') as f:
            history = pickle.load(f)
            
    return model, classes, history

with st.spinner("Initializing AI Backend Engine..."):
    model, class_names, history = load_backend_system()

# --- SIDEBAR NAVIGATION & INFO ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/doctor-male.png", width=70)
    st.markdown("### 🩺 DermaAI Pakistan Hub")
    
    st.markdown("---")
    selected_page = st.radio(
        "🧭 Navigation Menu:",
        ["🔬 Skin Cancer Detection & Referrals", "📊 Model Accuracy & Training Graphs"]
    )
    st.markdown("---")
    
    st.markdown("### 🇵🇰 Covered Regions")
    st.write("• Lahore\n• Karachi\n• Islamabad & Rawalpindi\n• Peshawar")
    
    st.markdown("---")
    st.markdown("<small>⚠️ *Disclaimer: AI tool designed for preliminary medical assistance. Physical clinical evaluation is imperative.*</small>", unsafe_allow_html=True)

# ==============================================================================
# PAGE 1: SKIN CANCER DETECTION & HOSPITAL REFERRALS
# ==============================================================================
if selected_page == "🔬 Skin Cancer Detection & Referrals":
    st.markdown("<h1 style='text-align: center; color: #1e3c72;'>🔬 DermaAI: Skin Cancer Detection & Referral</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #4f5d75; font-size: 17px;'>Upload a skin lesion image for instant neural network analysis and automated referrals to Pakistan's top-rated specialized doctors.</p>", unsafe_allow_html=True)
    st.markdown("---")

    col_upload, col_analysis = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📁 Lesion Image Input Panel")
        uploaded_file = st.file_uploader("Upload Skin Image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image_obj = Image.open(uploaded_file)
            st.image(image_obj, caption='Uploaded Target Lesion', use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_analysis:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("⚙️ Neural Network Inference Engine")
        
        if uploaded_file is not None:
            if st.button("🚀 Run AI Diagnostics & Check Referrals", use_container_width=True, type="primary"):
                with st.spinner("Analyzing image patterns against training weights..."):
                    img = image_obj.resize((224, 224))
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    preds = model.predict(img_array, verbose=0)
                    
                    if len(class_names) == 2:
                        score = float(preds[0][0])
                        probabilities = [1 - score, score]
                        if score >= 0.5:
                            detected_class = class_names[1]
                            confidence = score * 100
                            is_risk = True
                        else:
                            detected_class = class_names[0]
                            confidence = (1 - score) * 100
                            is_risk = False
                    else:
                        probabilities = preds[0].tolist()
                        idx = int(np.argmax(preds[0]))
                        detected_class = class_names[idx]
                        confidence = float(np.max(preds[0])) * 100
                        risk_
