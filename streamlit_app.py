import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="inSpecAI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load external CSS (assuming you have a 'styles.css' file)
load_css("styles.css")

# --- Configuration ---
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
IMAGE_SIZE = (224, 224)

# Custom DepthwiseConv2D layer for model loading compatibility
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, groups=None, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(*args, **kwargs)

# --- Helper Functions ---
@st.cache_resource
def load_my_model(path):
    try:
        model = tf.keras.models.load_model(
            path,
            custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D},
            compile=False
        )
        return model
    except Exception as e:
        st.error(f"❌ Neural Network Model Loading Failed: {e}")
        return None

@st.cache_data
def load_my_labels(labels_path):
    try:
        with open(labels_path, "r") as f:
            labels = [line.strip() for line in f.readlines()]
        return labels if labels else []
    except Exception as e:
        st.error(f"❌ Classification Labels Loading Failed: {e}")
        return []

def preprocess_image(image_data, target_size):
    try:
        if isinstance(image_data, str):
            image = Image.open(image_data)
        else:
            image = Image.open(image_data)
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = ImageOps.exif_transpose(image)
        image = image.resize(target_size)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, target_size[0], target_size[1], 3), dtype=np.float32)
        data[0] = normalized_image_array
        return data
    except Exception as e:
        st.error(f"⚠️ Image Processing Error: {e}")
        return None

def detect_defect(label):
    defect_keywords = [
        "damaged", "rusted", "cracked", "worn", "defect", "faulty", "bad",
        "broken", "corrupted", "scratch", "dent", "bend", "deformed",
        "missing", "misaligned", "discolored", "burnt"
    ]
    return any(keyword in label.lower() for keyword in defect_keywords)

def display_stat_card(value, label):
    st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{value}</div>
            <div class="stat-label">{label}</div>
        </div>
    """, unsafe_allow_html=True)

def display_result_card(status_text, classification, confidence, is_defective):
    if is_defective:
        card_class, status_indicator_class, status_icon, result_color = "result-error", "error", "🚨", "#ff6b6b"
    else:
        card_class, status_indicator_class, status_icon, result_color = "result-success", "success", "✅", "#4ecdc4"
        
    st.markdown(f"""
        <div class="result-card {card_class}">
            <div class="result-status" style="color: {result_color};">
                <span class="status-indicator {status_indicator_class}"></span>
                {status_icon} {status_text}
            </div>
            <div class="result-classification">
                Classification: <strong>{classification}</strong>
            </div>
            <div class="result-confidence">
                Confidence Level: {confidence*100:.1f}%
            </div>
        </div>
    """, unsafe_allow_html=True)

def create_confidence_chart(prediction, labels):
    confidences = [float(pred * 100) for pred in prediction[0]]
    fig = go.Figure(data=[go.Bar(
        y=labels, x=confidences, orientation='h',
        marker=dict(color=confidences, colorscale='Viridis', showscale=True, colorbar=dict(title="Confidence %"))
    )])
    fig.update_layout(
        title="Neural Network Confidence Analysis",
        xaxis_title="Confidence (%)", yaxis_title="Classifications",
        template="plotly_dark", height=400,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def save_inspection_log(result_data):
    if 'inspection_history' not in st.session_state:
        st.session_state.inspection_history = []
    
    st.session_state.inspection_history.insert(0, {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'classification': result_data['classification'],
        'confidence': result_data['confidence'],
        'status': result_data['status'],
        'is_defective': result_data['is_defective']
    })
    
    st.session_state.inspection_history = st.session_state.inspection_history[:100]

# --- Session State Initialization ---
if 'total_inspections' not in st.session_state:
    st.session_state.total_inspections = 0
if 'defects_detected' not in st.session_state:
    st.session_state.defects_detected = 0
if 'ai_confidence' not in st.session_state:
    st.session_state.ai_confidence = 99.2
if 'inspection_history' not in st.session_state:
    st.session_state.inspection_history = []
# MODIFICATION: Initialize state for persisting results
if 'last_analysis_result' not in st.session_state:
    st.session_state.last_analysis_result = None
if 'last_uploaded_file_id' not in st.session_state:
    st.session_state.last_uploaded_file_id = None


# Load model and labels
model = load_my_model(MODEL_PATH)
labels = load_my_labels(LABELS_PATH)

# --- Hero Section ---
# MODIFICATION: Wrapped the title in an anchor tag to make it clickable
st.markdown("""
    <div class="hero-container">
        <div class="hero-content">
            <a href="/" style="text-decoration: none; color: inherit;">
                <h1 class="hero-title">⚡ inSpecAI</h1>
            </a>
            <p class="hero-subtitle">FOR QC & ASSURANCE</p>
            <div class="hero-badge">🔬 AI-POWERED PRECISION ANALYSIS</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- Real-time Statistics Dashboard ---
if st.session_state.total_inspections > 0:
    pass_rate = ((st.session_state.total_inspections - st.session_state.defects_detected) / st.session_state.total_inspections) * 100
else:
    pass_rate = 0.0

st.markdown('<div class="stat-grid">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    display_stat_card(st.session_state.total_inspections, "Total Inspections")
with col2:
    display_stat_card(st.session_state.defects_detected, "Defects Found")
with col3:
    display_stat_card(f"{pass_rate:.1f}%", "Pass Rate")
with col4:
    display_stat_card(f"{st.session_state.ai_confidence:.1f}%", "AI Accuracy")
st.markdown("</div>", unsafe_allow_html=True)


if model is None or not labels:
    st.markdown("""
        <div class="neon-card">
            <h2 style="color: #ff6b6b; text-align: center;">⚠️ SYSTEM INITIALIZATION FAILED</h2>
            <p style="color: white; text-align: center; font-size: 1.1rem;">
                Neural network or classification parameters could not be loaded.<br>
                Please ensure 'keras_model.h5' and 'labels.txt' are properly configured and present.
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# --- Main Application Layout ---
main_tab1, main_tab2, main_tab3 = st.tabs(["🔍 INSPECTION", "📊 ANALYTICS", "📋 HISTORY"])

with main_tab1:
    main_col1, main_col2 = st.columns([3, 2])

    with main_col1:
        st.markdown("""
            <div class="neon-card" style="cursor: pointer; padding: 3rem; text-align: center; border: 2px dashed #667eea; border-radius: 20px; margin-bottom: 1rem;" id="upload-area">
                <h2 style="color: white; margin-bottom: 1rem;">🖼️ Component Upload Station</h2>
                <p style="color: #aaa;">Drag & drop an image here or click to browse files</p>
            </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            label="", type=["jpg", "jpeg", "png"],
            key="component_upload", label_visibility="collapsed"
        )

        # MODIFICATION: Logic to clear previous results when a new file is uploaded
        if uploaded_file and uploaded_file.file_id != st.session_state.last_uploaded_file_id:
            st.session_state.last_analysis_result = None
            st.session_state.last_uploaded_file_id = uploaded_file.file_id

        if uploaded_file:
            st.image(uploaded_file, caption="🔍 Component Ready for Analysis", use_container_width=True)

    with main_col2:
        st.markdown("""
            <div class="neon-card">
                <h2 style="color: white; margin-bottom: 1.5rem; text-align: center;">🤖 AI ANALYSIS CENTER</h2>
            </div>
        """, unsafe_allow_html=True)

        if uploaded_file:
            if st.button("🚀 INITIATE DEEP SCAN", type="primary", use_container_width=True):
                with st.spinner(""):
                    st.markdown("""
                        <div class="loading-container">
                            <div class="loading-spinner"></div>
                            <div class="loading-text">Neural Network Processing...</div>
                        </div>
                    """, unsafe_allow_html=True)
                    time.sleep(2)
                    
                    processed_image = preprocess_image(uploaded_file, IMAGE_SIZE)
                    
                    if processed_image is not None:
                        try:
                            prediction = model.predict(processed_image)
                            predicted_class_index = np.argmax(prediction[0])
                            predicted_label = labels[predicted_class_index]
                            confidence = float(prediction[0][predicted_class_index])
                            
                            st.session_state.total_inspections += 1
                            is_defective = detect_defect(predicted_label)
                            
                            if is_defective:
                                st.session_state.defects_detected += 1
                                status_text = "DEFECT DETECTED"
                            else:
                                status_text = "QUALITY APPROVED"
                            
                            st.session_state.ai_confidence = max(st.session_state.ai_confidence, confidence * 100)
                            
                            
                            # MODIFICATION: Store result in session state with the corrected key 'status'
                            st.session_state.last_analysis_result = {
                                'status': status_text, # <-- Corrected from 'status_text' to 'status'
                                'classification': predicted_label,
                                'confidence': confidence,
                                'is_defective': is_defective,
                                'prediction_array': prediction
                            }

                            save_inspection_log(st.session_state.last_analysis_result)
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"🔥 Neural Processing Error: {e}")

        # MODIFICATION: Display the stored result if it exists
        # MODIFICATION: Display the stored result using the corrected key 'status'
        if st.session_state.last_analysis_result:
            result = st.session_state.last_analysis_result
            display_result_card(
                result['status'], # <-- Corrected from 'status_text' to 'status'
                result['classification'],
                result['confidence'], 
                result['is_defective']
            )

            with st.expander("📊 DETAILED NEURAL ANALYSIS", expanded=True):
                fig = create_confidence_chart(result['prediction_array'], labels)
                st.plotly_chart(fig, use_container_width=True)
        elif uploaded_file:
             st.info("Ready for analysis. Press 'Initiate Deep Scan'.")
        else:
            st.markdown("""
                <div style="text-align: center; padding: 3rem; color: rgba(255, 255, 255, 0.6);">
                    <h3>⏳ AWAITING COMPONENT</h3>
                    <p>Upload an image to begin advanced quality analysis</p>
                </div>
            """, unsafe_allow_html=True)


with main_tab2:
    st.markdown("""
        <div class="neon-card">
            <h2 style="color: white; margin-bottom: 1.5rem; text-align: center;">📊 INSPECTION ANALYTICS</h2>
        </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.inspection_history:
        col1, col2 = st.columns(2)
        
        with col1:
            pass_count = len([h for h in st.session_state.inspection_history if not h['is_defective']])
            fail_count = len([h for h in st.session_state.inspection_history if h['is_defective']])
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Pass', 'Fail'], values=[pass_count, fail_count],
                marker=dict(colors=['#4ecdc4', '#ff6b6b'])
            )])
            fig_pie.update_layout(
                title="Pass/Fail Distribution", template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            confidences = [h['confidence'] * 100 for h in st.session_state.inspection_history]
            fig_hist = go.Figure(data=[go.Histogram(x=confidences, nbinsx=10, marker_color='#667eea')])
            fig_hist.update_layout(
                title="Confidence Distribution", xaxis_title="Confidence (%)", yaxis_title="Frequency",
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        if len(st.session_state.inspection_history) > 1:
            history_df = pd.DataFrame(st.session_state.inspection_history)
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            history_df = history_df.sort_values('timestamp')

            statuses = [0 if h else 1 for h in history_df['is_defective']]
            
            fig_timeline = go.Figure()
            fig_timeline.add_trace(go.Scatter(
                x=history_df['timestamp'], y=statuses, mode='lines+markers', name='Quality Status',
                line=dict(color='#4ecdc4'),
                marker=dict(size=8, symbol='circle', color=[('#ff6b6b' if s == 0 else '#4ecdc4') for s in statuses])
            ))
            fig_timeline.update_layout(
                title="Quality Status Timeline", xaxis_title="Time", yaxis=dict(tickvals=[0, 1], ticktext=['Defect', 'Pass']),
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.info("No inspection data available yet. Perform some inspections to see analytics.")

with main_tab3:
    st.markdown("""
        <div class="neon-card">
            <h2 style="color: white; margin-bottom: 1.5rem; text-align: center;">📋 INSPECTION HISTORY</h2>
        </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.inspection_history:
        history_df = pd.DataFrame(st.session_state.inspection_history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        
        col_export, col_clear = st.columns(2)
        with col_export:
            csv = history_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export to CSV", data=csv,
                file_name=f"inspection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv", use_container_width=True, type="secondary"
            )
        
        with col_clear:
            if st.button("🗑️ Clear History", type="secondary", use_container_width=True):
                st.session_state.inspection_history = []
                st.session_state.total_inspections = 0
                st.session_state.defects_detected = 0
                st.rerun()
    else:
        st.info("No inspection history available yet.")

with st.sidebar:
    st.markdown("""
        <div class="neon-card">
            <h2 style="color: white; margin-bottom: 1.5rem; text-align: center;">📹 LIVE INSPECTION FEED</h2>
        </div>
    """, unsafe_allow_html=True)
    
    camera_enabled = st.toggle("🔴 ACTIVATE LIVE CAMERA", value=False, help="Toggle to enable or disable live camera feed for inspection.")

    if camera_enabled:
        st.markdown("""
            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0; text-align: center;">
                <p style="color: white; font-weight: 500;">📱 Position component in camera view</p>
            </div>
        """, unsafe_allow_html=True)
        
        camera_image = st.camera_input("📸 CAPTURE FOR ANALYSIS", key="camera_input_live")

        if camera_image:
            with st.spinner("Analyzing live feed..."):
                processed_image = preprocess_image(camera_image, IMAGE_SIZE)
                if processed_image is not None:
                    try:
                        prediction = model.predict(processed_image)
                        predicted_class_index = np.argmax(prediction[0])
                        predicted_label = labels[predicted_class_index]
                        confidence = float(prediction[0][predicted_class_index])
                        is_defective = detect_defect(predicted_label)
                        
                        st.session_state.total_inspections += 1
                        if is_defective:
                            st.session_state.defects_detected += 1
                        
                        st.session_state.ai_confidence = max(st.session_state.ai_confidence, confidence * 100)

                        status_text = "DEFECT DETECTED" if is_defective else "QUALITY APPROVED"
                        
                        display_result_card(status_text, predicted_label, confidence, is_defective)

                        save_inspection_log({
                            'classification': predicted_label, 'confidence': confidence,
                            'status': status_text, 'is_defective': is_defective
                        })
                        
                        time.sleep(1) # a short delay to prevent rapid re-runs
                        st.rerun()

                    except Exception as e:
                        st.error(f"🔥 Live Analysis Error: {e}")
                else:
                    st.error("Failed to process live camera image.")
    else:
        st.info("Live camera is currently inactive.")