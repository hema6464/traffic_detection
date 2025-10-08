import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from ultralytics import YOLO
import os

# Page config
st.set_page_config(
    page_title="Traffic Sign Detection System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #e5e5e5;
    }
    .upload-box {
        border: 2px dashed #4a4a8a;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .upload-box:hover {
        border-color: #6a6aaa;
        background: rgba(255,255,255,0.08);
    }
    .detection-box {
        background: rgba(255,255,255,0.07);
        padding: 25px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metrics-card {
        background: rgba(255,255,255,0.1);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        transition: transform 0.3s ease;
    }
    .metrics-card:hover {
        transform: translateY(-5px);
    }
    .confidence-meter {
        height: 20px;
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        overflow: hidden;
        box-shadow: inset 0 0 5px rgba(0,0,0,0.2);
    }
    .header-container {
        padding: 20px;
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        margin-bottom: 20px;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(45deg, #3498db, #2980b9);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.total_analyses = 0

@st.cache_resource
def load_model():
    """Load and cache the YOLO model"""
    try:
        model = YOLO('best.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def create_confidence_bar(confidence):
    """Create a visual confidence meter"""
    colors = {
        "high": "#00ff00",
        "medium": "#ffaa00",
        "low": "#ff0000"
    }
    color = colors["high"] if confidence > 0.7 else colors["medium"] if confidence > 0.5 else colors["low"]
    return f"""
        <div class="confidence-meter">
            <div style="width:{confidence*100}%; height:100%; background:{color}; 
                 transition:width 0.5s; text-align:center; color:white; line-height:20px;">
                {confidence:.1%}
            </div>
        </div>
    """

def create_visualization(results, model):
    """Create visualization plots for detection results"""
    if len(results.boxes) > 0:
        # Prepare data for visualization
        confidences = results.boxes.conf.cpu().numpy()
        classes = [model.names[int(cls)] for cls in results.boxes.cls.cpu().numpy()]
        
        # Create confidence distribution plot
        fig_conf = go.Figure(data=[go.Histogram(x=confidences, nbinsx=20)])
        fig_conf.update_layout(
            title="Confidence Score Distribution",
            xaxis_title="Confidence Score",
            yaxis_title="Count",
            template="plotly_dark"
        )
        
        # Create class distribution plot
        class_counts = pd.Series(classes).value_counts()
        fig_class = px.pie(values=class_counts.values, names=class_counts.index,
                          title="Distribution of Detected Signs")
        fig_class.update_layout(template="plotly_dark")
        
        return fig_conf, fig_class
    return None, None

def main():
    # Header
    st.markdown("""
        <div class="header-container">
            <h1>🚦 Traffic Sign Detection System</h1>
            <p>Upload an image to detect and analyze traffic signs</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Analysis Settings")
        confidence_threshold = st.slider("Detection Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
        enable_analytics = st.checkbox("Enable Advanced Analytics", True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop your traffic sign image here or click to upload",
            type=['jpg', 'jpeg', 'png']
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
        st.markdown("### 📈 System Statistics")
        st.metric("Total Analyses", st.session_state.total_analyses)
        st.markdown("</div>", unsafe_allow_html=True)
    
    if uploaded_file:
        # Load and process image
        image_bytes = uploaded_file.read()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Original Image")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image_rgb)
        
        if st.button("🔍 Analyze Image", key="analyze_button"):
            st.session_state.total_analyses += 1
            
            with st.spinner("Processing image..."):
                # Load model if not already loaded
                if st.session_state.model is None:
                    st.session_state.model = load_model()
                
                if st.session_state.model is not None:
                    # Progress bar
                    progress_text = "Running analysis..."
                    progress_bar = st.progress(0)
                    # Run inference
                    results = st.session_state.model.predict(source=image, conf=confidence_threshold)[0]
                    
                    with col2:
                        st.markdown("### Detection Results")
                        plotted_image = results.plot()
                        plotted_image_rgb = cv2.cvtColor(plotted_image, cv2.COLOR_BGR2RGB)
                        st.image(plotted_image_rgb)
                    
                    # Display detections and analytics
                    if len(results.boxes) > 0:
                        st.markdown("<div class='detection-box'>", unsafe_allow_html=True)
                        st.markdown("### 🎯 Detection Details")
                        
                        for idx, box in enumerate(results.boxes):
                            class_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = st.session_state.model.names[class_id]
                            
                            with st.expander(f"Detection {idx+1}: {class_name} ({conf:.2%})"):
                                st.markdown(create_confidence_bar(conf), unsafe_allow_html=True)
                                box_coords = box.xyxy[0].cpu().numpy()
                                st.markdown(f"**Location:** x1={box_coords[0]:.1f}, y1={box_coords[1]:.1f}, x2={box_coords[2]:.1f}, y2={box_coords[3]:.1f}")
                        
                        if enable_analytics:
                            st.markdown("### 📊 Analytics")
                            fig_conf, fig_class = create_visualization(results, st.session_state.model)
                            if fig_conf and fig_class:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.plotly_chart(fig_conf, use_container_width=True)
                                with col2:
                                    st.plotly_chart(fig_class, use_container_width=True)
                        
                        # Summary metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
                            st.metric("Total Detections", len(results.boxes))
                            st.markdown("</div>", unsafe_allow_html=True)
                        with col2:
                            st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
                            avg_conf = float(torch.mean(results.boxes.conf).item())
                            st.metric("Average Confidence", f"{avg_conf:.2%}")
                            st.markdown("</div>", unsafe_allow_html=True)
                        with col3:
                            st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
                            unique_classes = len(set(results.boxes.cls.cpu().numpy()))
                            st.metric("Unique Sign Types", unique_classes)
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                    else:
                        st.warning("No traffic signs detected in this image.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Traffic Sign Detection System | Built with Streamlit and YOLOv8</p>
            <p>Deployed on Hugging Face Spaces | 2024</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Help section in sidebar
    with st.sidebar:
        st.markdown("---")
        with st.expander("ℹ️ Help & Instructions"):
            st.markdown("""
            ### How to Use:
            1. Upload an image containing traffic signs
            2. Adjust the confidence threshold if needed
            3. Click 'Analyze Image' to start detection
            4. View results and analytics
            
            ### Features:
            - Real-time traffic sign detection
            - Advanced analytics visualization
            - Confidence score analysis
            - Multiple sign detection support
            
            ### Tips:
            - Use clear, well-lit images
            - Adjust confidence threshold for better results
            - Enable analytics for detailed insights
            """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please refresh the page and try again.")

