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
        background: #f8f9fa;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 12px;
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
        {"hospital": "Shaukat Khanum Memorial Cancer Hospital", "location": "Johar Town, Lahore", "specialist": "Dr. Aasim Yusuf (Chief Medical Officer & Oncology)", "rating": "⭐ 4.9/5 (Top Oncology Center)", "contact": "042-35905000"},
        {"hospital": "Doctors Hospital & Medical Centre", "location": "Canal Bank, Lahore", "specialist": "Dr. Azam Samdani (Consultant Dermatologist)", "rating": "⭐ 4.8/5 (Skin Surgery Expert)", "contact": "042-111-223-311"},
        {"hospital": "Hamid Latif Hospital", "location": "Gulberg III, Lahore", "specialist": "Prof. Dr. Tahir Saeed Haroon (Dermatology)", "rating": "⭐ 4.7/5 (Senior Skin Specialist)", "contact": "042-35750817"}
    ],
    "Karachi": [
        {"hospital": "Aga Khan University Hospital (AKUH)", "location": "Stadium Road, Karachi", "specialist": "Dr. Zohra Zaidi (Consultant Dermatologist)", "rating": "⭐ 4.9/5 (Advanced Skin Diagnostics)", "contact": "021-111-911-911"},
        {"hospital": "Indus Hospital & Health Network", "location": "Korangi, Karachi", "specialist": "Oncology & Dermatological Surgery Unit", "rating": "⭐ 4.8/5 (Free Cancer Care & Screening)", "contact": "021-35112709"},
        {"hospital": "Ziauddin Hospital", "location": "Clifton, Karachi", "specialist": "Dr. Shahid Hassan (Senior Oncologist)", "rating": "⭐ 4.7/5 (Cancer Treatment Specialist)", "contact": "021-35862937"}
    ],
    "Islamabad / Rawalpindi": [
        {"hospital": "Shifa International Hospital", "location": "H-8/4, Islamabad", "specialist": "Dr. Humayun Mahmud (Consultant Dermatologist)", "rating": "⭐ 4.9/5 (Expert Skin Care & Laser)", "contact": "051-8464646"},
        {"hospital": "NORI Hospital (Nuclear Medicine & Oncology)", "location": "H-8/4, Islamabad", "specialist": "Dr. Muhammad Faheem (Head of Oncology)", "rating": "⭐ 4.8/5 (Government Cancer Specialized)", "contact": "051-9250553"},
        {"hospital": "Armed Forces Institute of Pathology (AFIP)", "location": "Rawalpindi", "specialist": "Department of Histopathology & Oncology", "rating": "⭐ 4.9/5 (Biopsy & Malignancy Confirmer)", "contact": "051-9271026"}
    ],
    "Peshawar": [
        {"hospital": "Northwest General Hospital & Research Centre", "location": "Hayatabad, Peshawar", "specialist": "Dr. Jawad Ahmad (Consultant Oncologist)", "rating": "⭐ 4.8/5 (Advanced Tumor Treatment)", "contact": "091-5838888"},
        {"hospital": "Hayatabad Medical Complex (HMC)", "location": "Hayatabad, Peshawar", "specialist": "Department of Dermatology & Oncology", "rating": "⭐ 4.6/5 (Public Healthcare Specialist)", "contact": "091-9217140"}
    ]
}

# --- LOAD BACKEND MODEL & CLASSES ---
@st.cache_resource
def load_backend_system():
    model = load_model('skin_cancer_model.h5')
    with open('class_names.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return model, classes

with st.spinner("Initializing AI Backend Engine..."):
    model, class_names = load_backend_system()

# --- SIDEBAR DESIGN ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/doctor-male.png", width=70)
    st.markdown("### 🩺 DermaAI Pakistan Hub")
    st.write("Nationwide automated referral system for skin cancer screening.")
    
    st.markdown("---")
    st.markdown("### 🇵🇰 Covered Regions")
    st.write("• Lahore\n• Karachi\n• Islamabad & Rawalpindi\n• Peshawar")
    
    st.markdown("---")
    st.markdown("<small>⚠️ *Disclaimer: AI tool designed for preliminary medical assistance. Physical clinical evaluation is imperative.*</small>", unsafe_allow_html=True)

# --- MAIN HEADER ---
st.markdown("<h1 style='text-align: center; color: #1e3c72;'>🔬 DermaAI: Pakistan Skin Cancer Detection & Referral</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #4f5d75; font-size: 17px;'>Upload a skin lesion image. If malignancy or risk is detected, the system will automatically refer you to Pakistan's top-rated specialized doctors and cancer hospitals with ratings.</p>", unsafe_allow_html=True)
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
                    risk_keywords = ['mel', 'bcc', 'akiec', 'malignant', 'cancer', 'carcinoma', 'melanoma']
                    is_risk = any(k in detected_class.lower() for k in risk_keywords)
                
                st.session_state['predicted_class'] = detected_class
                st.session_state['confidence'] = confidence
                st.session_state['is_risk'] = is_risk
                st.session_state['probabilities'] = probabilities
                st.session_state['analyzed'] = True

        if st.session_state.get('analyzed', False):
            d_class = st.session_state['predicted_class']
            conf = st.session_state['confidence']
            risk = st.session_state['is_risk']
            
            st.markdown(f"### 📌 Diagnosis Match: **{d_class.upper()}**")
            st.markdown(f"**Confidence Percentage:** `{conf:.2f}%`")
            st.progress(int(conf))
            
            if risk:
                st.markdown("""
                    <div class='danger-box'>
                    🚨 <strong>AUTOMATIC REFERRAL TRIGGERED:</strong> High malignancy / abnormal trait indicators detected. Please check the specialist directory below for immediate consultation.
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class='safe-box'>
                    ✅ <strong>Low Risk / Benign Indicator:</strong> The lesion aligns with low-risk features, but routine clinical observation is advised.
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.info("👈 Please upload an image on the left to activate scanning and specialist referrals.")
    st.markdown("</div>", unsafe_allow_html=True)

# --- CHARTS & PAKISTAN REFERRAL DIRECTORY SECTION ---
if st.session_state.get('analyzed', False):
    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>📊 Backend Analytics & Nationwide Specialist Referrals</h2>", unsafe_allow_html=True)
    
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📈 Probability Distribution Bar Chart")
        probs = st.session_state['probabilities']
        df_chart = pd.DataFrame({'Condition': class_names, 'Probability (%)': [p * 100 for p in probs]})
        
        # Fixed Plotly Bar Chart (No type mismatch error)
        fig = px.bar(df_chart, x='Condition', y='Probability (%)', color='Condition', text_auto='.2f')
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🍩 Diagnostic Ratio Breakdown")
        fig_pie = px.pie(df_chart, names='Condition', values='Probability (%)', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_pie.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- NATIONWIDE DOCTOR & HOSPITAL REFERRAL CARDS ---
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🏥 Official Referral Directory: Top Rated Doctors & Hospitals across Pakistan")
    st.write("Aap ya aapke mareez jis bhi shehar mein hain, neechay diye gaye top-rated cancer specialists aur hospitals se foran rabta karein:")
    
    city_tab = st.selectbox("Select Your City for Specialist Referral:", ["Lahore", "Karachi", "Islamabad / Rawalpindi", "Peshawar"])
    
    selected_docs = PAKISTAN_SPECIALISTS[city_tab]
    
    for doc in selected_docs:
        st.markdown(f"""
            <div class='doctor-card'>
                <h4>🏥 {doc['hospital']} ({doc['location']})</h4>
                <p><b>👨‍⚕️ Specialist / Expert:</b> {doc['specialist']}<br>
                <b>🌟 Rating & Profile:</b> {doc['rating']}<br>
                # --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/doctor-male.png", width=70)
    st.markdown("### 🩺 DermaAI Pakistan Hub")
    
    st.markdown("---")
    # Yeh raha sidebar ka alag navigation option:
    selected_page = st.radio(
        "🧭 Navigation Menu:",
        ["🔬 Skin Cancer Detection & Referrals", "📊 Model Accuracy & Training Graphs"]
    )
    st.markdown("---")
    
    st.markdown("### 🇵🇰 Covered Regions")
    st.write("• Lahore\n• Karachi\n• Islamabad & Rawalpindi\n• Peshawar")
                <b>📞 Helpline / Contact:</b> <a href="tel:{doc['contact']}">{doc['contact']}</a></p>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
