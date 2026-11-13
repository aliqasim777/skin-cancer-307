import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DermaAI - Advanced Skin Cancer Detection",
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
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(30, 60, 114, 0.3);
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 6px solid #dc3545;
        padding: 15px;
        border-radius: 6px;
        color: #721c24;
        font-weight: 500;
    }
    .safe-box {
        background-color: #d4edda;
        border-left: 6px solid #28a745;
        padding: 15px;
        border-radius: 6px;
        color: #155724;
        font-weight: 500;
    }
    h1, h2, h3 {
        color: #1a252f;
        font-family: 'Helvetica Neue', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD BACKEND MODEL & CLASSES ---
@st.cache_resource
def load_backend_system():
    model = load_model('skin_cancer_model.h5')
    with open('class_names.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return model, classes

with st.spinner("Initializing Backend AI Model & Loading Weights..."):
    model, class_names = load_backend_system()

# --- SIDEBAR DESIGN ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/doctor-male.png", width=70)
    st.markdown("### 🩺 DermaAI System Hub")
    st.write("Powered by MobileNetV2 Deep Learning & Transfer Learning Architecture.")
    
    st.markdown("---")
    st.markdown("### 🏥 Lahore Medical Network")
    st.success("**Shaukat Khanum Hospital**\nJohar Town, Lahore")
    st.info("**Doctors Hospital**\nCanal Bank, Lahore")
    st.info("**Hamid Latif Hospital**\nGulberg III, Lahore")
    
    st.markdown("---")
    st.markdown("<small>⚠️ *Disclaimer: Automated AI screening tool for preliminary diagnostics only. Physical verification by an oncologist or dermatologist is mandatory.*</small>", unsafe_allow_html=True)

# --- MAIN DASHBOARD HEADER ---
st.markdown("<h1 style='text-align: center; color: #1e3c72;'>🔬 DermaAI: Intelligent Skin Lesion Diagnostics</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #4f5d75; font-size: 17px;'>Upload any skin lesion photo (external or dataset) to run backend neural network analysis, percentage confidence tracking, and interactive probability charts.</p>", unsafe_allow_html=True)
st.markdown("---")

# --- LAYOUT COLUMNS ---
col_upload, col_analysis = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📁 Image Input Panel")
    uploaded_file = st.file_uploader("Upload Skin Lesion Image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image_obj = Image.open(uploaded_file)
        st.image(image_obj, caption='Target Lesion Preview', use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_analysis:
    st.markdown("<div class='card'>", unsafe_app_html=True)
    st.subheader("⚙️ Backend Neural Network Analysis")
    
    if uploaded_file is not None:
        if st.button("🚀 Run AI Diagnostics & Predictions", use_container_width=True, type="primary"):
            with st.spinner("Processing image tensor through CNN layers..."):
                # Preprocessing image for model
                img = image_obj.resize((224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) # Rescaling is built inside model graph
                
                # Model Prediction
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
                    risk_keywords = ['mel', 'bcc', 'akiec', 'malignant', 'cancer', 'carcinoma', 'melanoma']
                    is_risk = any(k in detected_class.lower() for k in risk_keywords)
                
                # Session State Storage for Charts
                st.session_state['predicted_class'] = detected_class
                st.session_state['confidence'] = confidence
                st.session_state['is_risk'] = is_risk
                st.session_state['probabilities'] = probabilities
                st.session_state['analyzed'] = True

        # Display Results if analyzed
        if st.session_state.get('analyzed', False):
            d_class = st.session_state['predicted_class']
            conf = st.session_state['confidence']
            risk = st.session_state['is_risk']
            
            st.markdown(f"### 📌 Diagnosis: **{d_class.upper()}**")
            st.markdown(f"**Confidence Match Percentage:** `{conf:.2f}%`")
            st.progress(int(conf))
            
            if risk:
                st.markdown("""
                    <div class='danger-box'>
                    ⚠️ <strong>High Malignancy Risk Detected:</strong> The backend model registers high probabilities for abnormal lesion traits. Immediate consultation with a specialist is strongly advised.
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class='safe-box'>
                    ✅ <strong>Low Risk / Benign Indicator:</strong> The structural features align with low-risk skin categories. Routine clinical observation is recommended.
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.info("👈 Please upload an image on the left panel to execute backend inference.")
    st.markdown("</div>", unsafe_allow_html=True)

# --- ADVANCED GRAPHS & CHARTS SECTION (FULL WIDTH) ---
if st.session_state.get('analyzed', False):
    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>📊 Backend Probability Distribution & Analytics Charts</h2>", unsafe_allow_html=True)
    
    chart_col1, chart_col2 = st.columns([1, 1], gap="large")
    
    with chart_col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📈 Class Probability Bar Chart")
        probs = st.session_state['probabilities']
        
        # Creating a neat dataframe for Plotly
        df_chart = pd.DataFrame({
            'Skin Condition': class_names,
            'Probability (%)': [p * 100 for p in probs]
        })
        
        fig = px.bar(
            df_chart, 
            x='Skin Condition', 
            y='Probability (%)', 
            color='Probability (%)',
            color_continuousscale='Viridis',
            text_auto='.2f'
        )
        fig.update_layout(title="Confidence Score Across All Classes", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with chart_col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🍩 Diagnostic Confidence Breakdown")
        
        fig_pie = px.pie(
            df_chart, 
            names='Skin Condition', 
            values='Probability (%)',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_pie.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- DOCTOR & HOSPITAL RECOMMENDATION CARD ---
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🏥 Recommended Medical Specialists & Hospitals in Lahore")
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.markdown("**Shaukat Khanum Hospital**")
        st.write("📍 Johar Town, Lahore")
        st.write("🔬 Specialized Oncology & Biopsy Screening")
    with col_d2:
        st.markdown("**Doctors Hospital**")
        st.write("📍 Canal Bank, Lahore")
        st.write("👨‍⚕️ Senior Consultant Dermatologists")
    with col_d3:
        st.markdown("**Hamid Latif Hospital**")
        st.write("📍 Gulberg III, Lahore")
        st.write("💡 Advanced Laser & Skin Surgery")
    st.markdown("</div>", unsafe_allow_html=True)
