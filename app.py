
# import streamlit as st
# from PIL import Image
# import torch
# import cv2
# import numpy as np
# from datetime import datetime
# import plotly.express as px
# import plotly.graph_objects as go
# import pandas as pd
# from ultralytics import YOLO
# import os
# import io
# from fpdf import FPDF
# import base64

# # Page config
# st.set_page_config(
#     page_title="Traffic Sign Detection System",
#     page_icon="🚦",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS
# st.markdown("""
#     <style>
#     .stApp {
#         background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
#         color: #e5e5e5;
#     }
#     .upload-box {
#         border: 2px dashed #4a4a8a;
#         border-radius: 15px;
#         padding: 30px;
#         text-align: center;
#         background: rgba(255,255,255,0.05);
#         backdrop-filter: blur(10px);
#         transition: all 0.3s ease;
#     }
#     .upload-box:hover {
#         border-color: #6a6aaa;
#         background: rgba(255,255,255,0.08);
#     }
#     .detection-box {
#         background: rgba(255,255,255,0.07);
#         padding: 25px;
#         border-radius: 15px;
#         margin: 15px 0;
#         box-shadow: 0 4px 15px rgba(0,0,0,0.2);
#     }
#     .metrics-card {
#         background: rgba(255,255,255,0.1);
#         padding: 20px;
#         border-radius: 10px;
#         margin: 10px 0;
#         transition: transform 0.3s ease;
#     }
#     .metrics-card:hover {
#         transform: translateY(-5px);
#     }
#     .confidence-meter {
#         height: 20px;
#         background: rgba(255,255,255,0.1);
#         border-radius: 10px;
#         overflow: hidden;
#         box-shadow: inset 0 0 5px rgba(0,0,0,0.2);
#     }
#     .header-container {
#         padding: 20px;
#         background: rgba(255,255,255,0.05);
#         border-radius: 15px;
#         margin-bottom: 20px;
#         text-align: center;
#     }
#     .stButton>button {
#         background: linear-gradient(45deg, #3498db, #2980b9);
#         color: white;
#         border: none;
#         padding: 10px 20px;
#         border-radius: 8px;
#         transition: all 0.3s ease;
#     }
#     .stButton>button:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 5px 15px rgba(0,0,0,0.2);
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Initialize session state
# if 'model' not in st.session_state:
#     st.session_state.model = None
#     st.session_state.total_analyses = 0

# @st.cache_resource
# def load_model():
#     """Load and cache the YOLO model"""
#     try:
#         model = YOLO('best.pt')
#         return model
#     except Exception as e:
#         st.error(f"Error loading model: {str(e)}")
#         return None

# def create_confidence_bar(confidence):
#     """Create a visual confidence meter"""
#     colors = {
#         "high": "#00ff00",
#         "medium": "#ffaa00",
#         "low": "#ff0000"
#     }
#     color = colors["high"] if confidence > 0.7 else colors["medium"] if confidence > 0.5 else colors["low"]
#     return f"""
#         <div class="confidence-meter">
#             <div style="width:{confidence*100}%; height:100%; background:{color}; 
#                  transition:width 0.5s; text-align:center; color:white; line-height:20px;">
#                 {confidence:.1%}
#             </div>
#         </div>
#     """

# def create_visualization(results, model):
#     """Create visualization plots for detection results"""
#     if len(results.boxes) > 0:
#         # Prepare data for visualization
#         confidences = results.boxes.conf.cpu().numpy()
#         classes = [model.names[int(cls)] for cls in results.boxes.cls.cpu().numpy()]
        
#         # Create confidence distribution plot
#         fig_conf = go.Figure(data=[go.Histogram(x=confidences, nbinsx=20)])
#         fig_conf.update_layout(
#             title="Confidence Score Distribution",
#             xaxis_title="Confidence Score",
#             yaxis_title="Count",
#             template="plotly_dark"
#         )
        
#         # Create class distribution plot
#         class_counts = pd.Series(classes).value_counts()
#         fig_class = px.pie(values=class_counts.values, names=class_counts.index,
#                           title="Distribution of Detected Signs")
#         fig_class.update_layout(template="plotly_dark")
        
#         return fig_conf, fig_class
#     return None, None

# class PDF(FPDF):
#     def header(self):
#         self.set_font('Arial', 'B', 20)
#         self.cell(0, 10, 'Traffic Sign Detection Report', 0, 1, 'C')
#         self.ln(10)
        
#     def footer(self):
#         self.set_y(-15)
#         self.set_font('Arial', 'I', 8)
#         self.cell(0, 10, f'Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 0, 'C')

# def generate_pdf_report(original_image, detected_image, results, model, analytics_enabled=True):
#     """Generate a PDF report with detection results and analytics"""
#     pdf = PDF()
#     pdf.add_page()
    
#     # Add timestamp
#     pdf.set_font('Arial', '', 12)
#     pdf.cell(0, 10, f'Analysis Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1)
#     pdf.ln(5)
    
#     # Save and add images
#     # Original Image
#     pdf.set_font('Arial', 'B', 14)
#     pdf.cell(0, 10, 'Original Image', 0, 1)
    
#     # Convert original image to RGB if needed
#     if len(original_image.shape) == 3 and original_image.shape[2] == 3:
#         original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
#     else:
#         original_image_rgb = original_image
        
#     # Save original image
#     orig_img_path = 'temp_original.jpg'
#     cv2.imwrite(orig_img_path, cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2BGR))
#     pdf.image(orig_img_path, x=10, w=190)
#     pdf.ln(10)
    
#     # Detection Results
#     pdf.set_font('Arial', 'B', 14)
#     pdf.cell(0, 10, 'Detection Results', 0, 1)
    
#     # Save detected image
#     detected_img_path = 'temp_detected.jpg'
#     cv2.imwrite(detected_img_path, cv2.cvtColor(detected_image, cv2.COLOR_RGB2BGR))
#     pdf.image(detected_img_path, x=10, w=190)
#     pdf.ln(10)
    
#     # Detection Details
#     pdf.set_font('Arial', 'B', 14)
#     pdf.cell(0, 10, 'Detection Details', 0, 1)
#     pdf.set_font('Arial', '', 12)
    
#     if len(results.boxes) > 0:
#         # Summary statistics
#         avg_conf = float(torch.mean(results.boxes.conf).item())
#         unique_classes = len(set(results.boxes.cls.cpu().numpy()))
        
#         pdf.cell(0, 10, f'Total Detections: {len(results.boxes)}', 0, 1)
#         pdf.cell(0, 10, f'Average Confidence: {avg_conf:.2%}', 0, 1)
#         pdf.cell(0, 10, f'Unique Sign Types: {unique_classes}', 0, 1)
#         pdf.ln(5)
        
#         # Detailed detections
#         pdf.set_font('Arial', 'B', 12)
#         pdf.cell(0, 10, 'Detailed Detections:', 0, 1)
#         pdf.set_font('Arial', '', 12)
        
#         for idx, box in enumerate(results.boxes):
#             class_id = int(box.cls[0])
#             conf = float(box.conf[0])
#             class_name = model.names[class_id]
#             box_coords = box.xyxy[0].cpu().numpy()
            
#             pdf.cell(0, 10, f'Detection {idx+1}:', 0, 1)
#             pdf.cell(0, 10, f'  • Sign Type: {class_name}', 0, 1)
#             pdf.cell(0, 10, f'  • Confidence: {conf:.2%}', 0, 1)
#             pdf.cell(0, 10, f'  • Location: x1={box_coords[0]:.1f}, y1={box_coords[1]:.1f}, x2={box_coords[2]:.1f}, y2={box_coords[3]:.1f}', 0, 1)
#             pdf.ln(5)
#     else:
#         pdf.cell(0, 10, 'No traffic signs detected in this image.', 0, 1)
    
#     # Clean up temporary files
#     try:
#         os.remove(orig_img_path)
#         os.remove(detected_img_path)
#     except:
#         pass
    
#     # Save to bytes
#     pdf_bytes = io.BytesIO()
#     pdf.output(pdf_bytes)
#     pdf_bytes.seek(0)
    
#     return pdf_bytes

# def main():
#     # Header
#     st.markdown("""
#         <div class="header-container">
#             <h1>🚦 Traffic Sign Detection System</h1>
#             <p>Upload an image to detect and analyze traffic signs</p>
#         </div>
#     """, unsafe_allow_html=True)
    
#     # Sidebar
#     with st.sidebar:
#         st.header("📊 Analysis Settings")
#         # confidence_threshold = st.slider("Detection Confidence Threshold", 0.0, 1.0, 0.25, 0.05) = st.slider("Detection Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

#         confidence_threshold = st.slider("Detection Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
#         enable_analytics = st.checkbox("Enable Advanced Analytics", True)
    
#     # Main content
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
#         uploaded_file = st.file_uploader(
#             "Drop your traffic sign image here or click to upload",
#             type=['jpg', 'jpeg', 'png']
#         )
#         st.markdown("</div>", unsafe_allow_html=True)
    
#     with col2:
#         st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
#         st.markdown("### 📈 System Statistics")
#         st.metric("Total Analyses", st.session_state.total_analyses)
#         st.markdown("</div>", unsafe_allow_html=True)
    
#     if uploaded_file:
#         # Load and process image
#         image_bytes = uploaded_file.read()
#         image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.markdown("### Original Image")
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             st.image(image_rgb)
        
#         if st.button("🔍 Analyze Image", key="analyze_button"):
#             st.session_state.total_analyses += 1
            
#             with st.spinner("Processing image..."):
#                 # Load model if not already loaded
#                 if st.session_state.model is None:
#                     st.session_state.model = load_model()
                
#                 if st.session_state.model is not None:
#                     # Progress bar
#                     progress_text = "Running analysis..."
#                     progress_bar = st.progress(0)
#                     # for i in range(100):
#                     #     progress_bar.progress(i + 1)
#                     #     if i == 30:
#                     #         progress_text = "Detecting signs..."
#                     #     elif i == 60:
#                     #         progress_text = "Analyzing patterns..."
#                     #     elif i == 90:
#                     #         progress_text = "Preparing results..."
#                     #     st.write(progress_text)
                    
#                     # Run inference
#                     results = st.session_state.model.predict(source=image, conf=confidence_threshold)[0]
                    
#                     with col2:
#                         st.markdown("### Detection Results")
#                         plotted_image = results.plot()
#                         plotted_image_rgb = cv2.cvtColor(plotted_image, cv2.COLOR_BGR2RGB)
#                         st.image(plotted_image_rgb)
                    
#                     # Display detections and analytics
#                     if len(results.boxes) > 0:
#                         st.markdown("<div class='detection-box'>", unsafe_allow_html=True)
#                         st.markdown("### 🎯 Detection Details")
                        
#                         for idx, box in enumerate(results.boxes):
#                             class_id = int(box.cls[0])
#                             conf = float(box.conf[0])
#                             class_name = st.session_state.model.names[class_id]
                            
#                             with st.expander(f"Detection {idx+1}: {class_name} ({conf:.2%})"):
#                                 st.markdown(create_confidence_bar(conf), unsafe_allow_html=True)
#                                 box_coords = box.xyxy[0].cpu().numpy()
#                                 st.markdown(f"**Location:** x1={box_coords[0]:.1f}, y1={box_coords[1]:.1f}, x2={box_coords[2]:.1f}, y2={box_coords[3]:.1f}")
                        
#                         if enable_analytics:
#                             st.markdown("### 📊 Analytics")
#                             fig_conf, fig_class = create_visualization(results, st.session_state.model)
#                             if fig_conf and fig_class:
#                                 col1, col2 = st.columns(2)
#                                 with col1:
#                                     st.plotly_chart(fig_conf, use_container_width=True)
#                                 with col2:
#                                     st.plotly_chart(fig_class, use_container_width=True)
                        
#                         # Summary metrics
#                         col1, col2, col3 = st.columns(3)
#                         with col1:
#                             st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
#                             st.metric("Total Detections", len(results.boxes))
#                             st.markdown("</div>", unsafe_allow_html=True)
#                         with col2:
#                             st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
#                             # avg_conf = float(torch.mean(results.boxes
#                             avg_conf = float(torch.mean(results.boxes.conf).item())
#                             st.metric("Average Confidence", f"{avg_conf:.1%}")
#                             st.markdown("</div>", unsafe_allow_html=True)
#                         with col3:
#                             st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
#                             unique_classes = len(set(results.boxes.cls.cpu().numpy()))
#                             st.metric("Unique Sign Types", unique_classes)
#                             st.markdown("</div>", unsafe_allow_html=True)
                        
#                         # Generate and offer PDF report download
#                         st.markdown("### 📄 Download Report")
#                         pdf_bytes = generate_pdf_report(
#                             image,
#                             plotted_image,
#                             results,
#                             st.session_state.model,
#                             analytics_enabled=enable_analytics
#                         )
                        
#                         # Create download button with custom styling
#                         st.markdown("""
#                             <style>
#                             .download-button {
#                                 background: linear-gradient(45deg, #2ecc71, #27ae60);
#                                 color: white;
#                                 padding: 12px 24px;
#                                 border-radius: 8px;
#                                 border: none;
#                                 cursor: pointer;
#                                 transition: all 0.3s ease;
#                                 text-decoration: none;
#                                 display: inline-block;
#                                 margin: 10px 0;
#                             }
#                             .download-button:hover {
#                                 transform: translateY(-2px);
#                                 box-shadow: 0 5px 15px rgba(0,0,0,0.2);
#                             }
#                             </style>
#                         """, unsafe_allow_html=True)
                        
#                         # Convert PDF bytes to base64 for download
#                         b64_pdf = base64.b64encode(pdf_bytes.getvalue()).decode()
#                         href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="traffic_sign_report.pdf" class="download-button">📥 Download Detection Report</a>'
#                         st.markdown(href, unsafe_allow_html=True)
                        
#                         st.markdown("</div>", unsafe_allow_html=True)
#                     else:
#                         st.warning("No traffic signs detected in the image.")
#                 else:
#                     st.error("Failed to load the model. Please check if the model file exists and try again.")


# if __name__ == "__main__":
#     try:
#         main()
#     except Exception as e:
#         st.error(f"An error occurred: {str(e)}")
#         st.error("Please refresh the page and try again.")                           












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
                    # for i in range(100):
                    #     progress_bar.progress(i + 1)
                    #     if i == 30:
                    #         progress_text = "Detecting signs..."
                    #     elif i == 60:
                    #         progress_text = "Analyzing patterns..."
                    #     elif i == 90:
                    #         progress_text = "Preparing results..."
                    #     st.write(progress_text)
                    
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
















# import streamlit as st
# from PIL import Image
# import torch
# import cv2
# import numpy as np
# from datetime import datetime
# import plotly.express as px
# import plotly.graph_objects as go
# import io
# import base64
# from fpdf import FPDF
# import os
# from huggingface_hub import hf_hub_download

# # Page config
# st.set_page_config(
#     page_title="Traffic Sign Detection System",
#     page_icon="🚦",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS with improved styling
# st.markdown("""
#     <style>
#     .stApp {
#         background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
#         color: #e5e5e5;
#     }
#     .upload-box {
#         border: 2px dashed #4a4a8a;
#         border-radius: 15px;
#         padding: 30px;
#         text-align: center;
#         background: rgba(255,255,255,0.05);
#         backdrop-filter: blur(10px);
#         transition: all 0.3s ease;
#     }
#     .upload-box:hover {
#         border-color: #6a6aaa;
#         background: rgba(255,255,255,0.08);
#     }
#     .detection-box {
#         background: rgba(255,255,255,0.07);
#         padding: 25px;
#         border-radius: 15px;
#         margin: 15px 0;
#         box-shadow: 0 4px 15px rgba(0,0,0,0.2);
#     }
#     .metrics-card {
#         background: rgba(255,255,255,0.1);
#         padding: 20px;
#         border-radius: 10px;
#         margin: 10px 0;
#         transition: transform 0.3s ease;
#     }
#     .metrics-card:hover {
#         transform: translateY(-5px);
#     }
#     .confidence-meter {
#         height: 20px;
#         background: rgba(255,255,255,0.1);
#         border-radius: 10px;
#         overflow: hidden;
#         box-shadow: inset 0 0 5px rgba(0,0,0,0.2);
#     }
#     .header-container {
#         padding: 20px;
#         background: rgba(255,255,255,0.05);
#         border-radius: 15px;
#         margin-bottom: 20px;
#         text-align: center;
#     }
#     .stButton>button {
#         background: linear-gradient(45deg, #3498db, #2980b9);
#         color: white;
#         border: none;
#         padding: 10px 20px;
#         border-radius: 8px;
#         transition: all 0.3s ease;
#     }
#     .stButton>button:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 5px 15px rgba(0,0,0,0.2);
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Initialize session state
# if 'model' not in st.session_state:
#     st.session_state.model = None
#     st.session_state.total_analyses = 0

# @st.cache_resource
# def load_model():
#     """Load and cache the YOLO model from Hugging Face Hub"""
#     try:
#         # Download model from Hugging Face Hub
#         model_path = hf_hub_download(repo_id="YOUR_HUGGINGFACE_USERNAME/traffic-sign-detection", 
#                                    filename="best.pt")
#         return torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
#     except Exception as e:
#         st.error(f"Error loading model: {str(e)}")
#         return None

# def create_confidence_bar(confidence):
#     """Create a visual confidence meter"""
#     colors = {
#         "high": "#00ff00",
#         "medium": "#ffaa00",
#         "low": "#ff0000"
#     }
#     color = colors["high"] if confidence > 0.7 else colors["medium"] if confidence > 0.5 else colors["low"]
#     return f"""
#         <div class="confidence-meter">
#             <div style="width:{confidence*100}%; height:100%; background:{color}; 
#                  transition:width 0.5s; text-align:center; color:white; line-height:20px;">
#                 {confidence:.1%}
#             </div>
#         </div>
#     """

# def create_visualization(results):
#     """Create visualization plots for detection results"""
#     if len(results.xyxy[0]) > 0:
#         # Prepare data for visualization
#         confidences = results.xyxy[0][:, 4].cpu().numpy()
#         classes = [results.names[int(cls)] for cls in results.xyxy[0][:, 5].cpu().numpy()]
        
#         # Create confidence distribution plot
#         fig_conf = go.Figure(data=[go.Histogram(x=confidences, nbinsx=20)])
#         fig_conf.update_layout(
#             title="Confidence Score Distribution",
#             xaxis_title="Confidence Score",
#             yaxis_title="Count",
#             template="plotly_dark"
#         )
        
#         # Create class distribution plot
#         class_counts = pd.Series(classes).value_counts()
#         fig_class = px.pie(values=class_counts.values, names=class_counts.index,
#                           title="Distribution of Detected Signs")
#         fig_class.update_layout(template="plotly_dark")
        
#         return fig_conf, fig_class
#     return None, None

# def main():
#     # Header
#     st.markdown("""
#         <div class="header-container">
#             <h1>🚦 Traffic Sign Detection System</h1>
#             <p>Upload an image to detect and analyze traffic signs</p>
#         </div>
#     """, unsafe_allow_html=True)
    
#     # Sidebar
#     with st.sidebar:
#         st.header("📊 Analysis Settings")
#         confidence_threshold = st.slider("Detection Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
#         enable_analytics = st.checkbox("Enable Advanced Analytics", True)
    
#     # Main content
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
#         uploaded_file = st.file_uploader(
#             "Drop your traffic sign image here or click to upload",
#             type=['jpg', 'jpeg', 'png']
#         )
#         st.markdown("</div>", unsafe_allow_html=True)
    
#     if uploaded_file:
#         # Load and process image
#         image = Image.open(uploaded_file)
#         img_array = np.array(image)
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.markdown("### Original Image")
#             st.image(image)
        
#         if st.button("🔍 Analyze Image", key="analyze_button"):
#             st.session_state.total_analyses += 1
            
#             with st.spinner("Processing image..."):
#                 # Load model if not already loaded
#                 if st.session_state.model is None:
#                     st.session_state.model = load_model()
                
#                 if st.session_state.model is not None:
#                     # Run inference
#                     results = st.session_state.model(img_array)
                    
#                     with col2:
#                         st.markdown("### Detection Results")
#                         img_with_boxes = results.render()[0]
#                         st.image(img_with_boxes)
                    
#                     # Display detections and analytics
#                     if len(results.xyxy[0]) > 0:
#                         st.markdown("### 🎯 Detection Details")
#                         for idx, detection in enumerate(results.xyxy[0]):
#                             conf = float(detection[4])
#                             class_id = int(detection[5])
#                             class_name = results.names[class_id]
                            
#                             with st.expander(f"Detection {idx+1}: {class_name} ({conf:.2%})"):
#                                 st.markdown(create_confidence_bar(conf), unsafe_allow_html=True)
#                                 st.markdown(f"**Location:** x1={detection[0]:.1f}, y1={detection[1]:.1f}, x2={detection[2]:.1f}, y2={detection[3]:.1f}")
                        
#                         if enable_analytics:
#                             st.markdown("### 📊 Analytics")
#                             fig_conf, fig_class = create_visualization(results)
#                             if fig_conf and fig_class:
#                                 col1, col2 = st.columns(2)
#                                 with col1:
#                                     st.plotly_chart(fig_conf, use_container_width=True)
#                                 with col2:
#                                     st.plotly_chart(fig_class, use_container_width=True)
#                     else:
#                         st.warning("No traffic signs detected in this image.")
    
#     # Footer
#     st.markdown("---")
#     st.markdown(
#         """
#         <div style='text-align: center'>
#             <p>Traffic Sign Detection System | Built with Streamlit and YOLOv5</p>
#             <p>Deployed on Hugging Face Spaces | 2024</p>
#         </div>
#         """, 
#         unsafe_allow_html=True
#     )

# if __name__ == "__main__":
#     try:
#         main()
#     except Exception as e:
#         st.error(f"An error occurred: {str(e)}")
#         st.error("Please refresh the page and try again.")



















# # # import streamlit as st
# # # from ultralytics import YOLO
# # # import cv2
# # # import numpy as np
# # # from PIL import Image
# # # import time

# # # # Page config
# # # st.set_page_config(
# # #     page_title="Traffic Sign Detection",
# # #     page_icon="🚦",
# # #     layout="wide"
# # # )

# # # # Custom CSS
# # # st.markdown("""
# # #     <style>
# # #     .stApp {
# # #         background: linear-gradient(to right, #1a1a1a, #2d2d2d);
# # #         color: white;
# # #     }
# # #     .upload-box {
# # #         border: 2px dashed #4a4a4a;
# # #         border-radius: 10px;
# # #         padding: 20px;
# # #         text-align: center;
# # #         background: rgba(255,255,255,0.05);
# # #     }
# # #     .detection-box {
# # #         background: rgba(255,255,255,0.1);
# # #         padding: 20px;
# # #         border-radius: 10px;
# # #         margin: 10px 0;
# # #     }
# # #     .confidence-meter {
# # #         height: 20px;
# # #         background: green;
# # #         border-radius: 10px;
# # #         overflow: hidden;
# # #     }
# # #     </style>
# # # """, unsafe_allow_html=True)

# # # @st.cache_resource
# # # def load_model():
# # #     return YOLO("best.pt")

# # # def create_confidence_bar(confidence):
# # #     color = "green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"
# # #     return f"""
# # #         <div class="confidence-meter">
# # #             <div style="width:{confidence*100}%; height:100%; background:{color}; transition:width 0.5s;">
# # #             </div>
# # #         </div>
# # #     """

# # # def main():
# # #     # Header
# # #     col1, col2, col3 = st.columns([1,2,1])
# # #     with col2:
# # #         st.title("🚦 Traffic Sign Detection")
# # #         st.markdown("### Advanced AI-Powered Traffic Sign Recognition")
    
# # #     # Model loading with spinner
# # #     with st.spinner("Loading AI Model..."):
# # #         model = load_model()
    
# # #     # File upload section
# # #     st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
# # #     uploaded_file = st.file_uploader(
# # #         "Drop your image here or click to upload",
# # #         type=['jpg', 'jpeg', 'png']
# # #     )
# # #     st.markdown("</div>", unsafe_allow_html=True)
    
# # #     if uploaded_file:
# # #         # Image processing
# # #         image_bytes = uploaded_file.read()
# # #         image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
# # #         # Display columns
# # #         col1, col2 = st.columns(2)
        
# # #         with col1:
# # #             st.markdown("### Original Image")
# # #             st.image(image, channels="BGR", use_container_width=True)
        
# # #         # Process button
# # #         if st.button("🔍 Analyze Image", use_container_width=True):
# # #             with st.spinner("Analyzing image..."):
# # #                 # Add progress bar
# # #                 progress_bar = st.progress(0)
# # #                 for i in range(100):
# # #                     time.sleep(0.01)
# # #                     progress_bar.progress(i + 1)
                
# # #                 # Inference
# # #                 image_resized = cv2.resize(image, (640, 640))
# # #                 results = model.predict(source=image_resized, conf=0.25)[0]
# # #                 plotted_image = results.plot()
                
# # #                 with col2:
# # #                     st.markdown("### Detection Results")
# # #                     st.image(plotted_image, channels="BGR", use_container_width=True)
                
# # #                 # Results section
# # #                 st.markdown("<div class='detection-box'>", unsafe_allow_html=True)
# # #                 st.markdown("### 📊 Detailed Analysis")
                
# # #                 if len(results.boxes) > 0:
# # #                     for idx, box in enumerate(results.boxes):
# # #                         class_id = box.cls.cpu().numpy()[0]
# # #                         conf = box.conf.cpu().numpy()[0]
# # #                         class_name = model.names[int(class_id)]
                        
# # #                         # Create expandable section for each detection
# # #                         with st.expander(f"Detection {idx+1}: {class_name}"):
# # #                             st.markdown(f"**Confidence Score:** {conf:.2%}")
# # #                             st.markdown(create_confidence_bar(conf), unsafe_allow_html=True)
                            
# # #                             # Box coordinates
# # #                             box_coords = box.xyxy.cpu().numpy()[0]
# # #                             st.markdown("**Location Details:**")
# # #                             st.code(f"X1: {box_coords[0]:.1f}, Y1: {box_coords[1]:.1f}\nX2: {box_coords[2]:.1f}, Y2: {box_coords[3]:.1f}")
# # #                 else:
# # #                     st.warning("No traffic signs detected in this image.")
                
# # #                 st.markdown("</div>", unsafe_allow_html=True)
                
# # #                 # Summary metrics
# # #                 col1, col2, col3 = st.columns(3)
# # #                 with col1:
# # #                     st.metric("Total Detections", len(results.boxes))
# # #                 with col2:
# # #                     avg_conf = np.mean(results.boxes.conf.cpu().numpy()) if len(results.boxes) > 0 else 0
# # #                     st.metric("Average Confidence", f"{avg_conf:.2%}")
# # #                 with col3:
# # #                     unique_classes = len(set(results.boxes.cls.cpu().numpy())) if len(results.boxes) > 0 else 0
# # #                     st.metric("Unique Sign Types", unique_classes)

# # #     # Footer
# # #     st.markdown("---")
# # #     st.markdown(
# # #         "Made with ❤️ using YOLOv8 and Streamlit | "
# # #         "[GitHub](https://github.com) | "
# # #         "[Report Issue](https://github.com/issues)"
# # #     )

# # # if __name__ == "__main__":
# # #     main()







# # # # import streamlit as st
# # # # from ultralytics import YOLO
# # # # import cv2
# # # # import numpy as np
# # # # from PIL import Image
# # # # import time
# # # # from fpdf import FPDF
# # # # import io
# # # # import base64

# # # # st.set_page_config(page_title="Traffic Sign Detection", page_icon="🚦", layout="wide")

# # # # st.markdown("""
# # # #     <style>
# # # #     .stApp {
# # # #         background: linear-gradient(to right, #1a1a1a, #2d2d2d);
# # # #         color: white;
# # # #     }
# # # #     .upload-box {
# # # #         border: 2px dashed #4a4a4a;
# # # #         border-radius: 10px;
# # # #         padding: 20px;
# # # #         text-align: center;
# # # #         background: rgba(255,255,255,0.05);
# # # #     }
# # # #     .detection-box {
# # # #         background: rgba(255,255,255,0.1);
# # # #         padding: 20px;
# # # #         border-radius: 10px;
# # # #         margin: 10px 0;
# # # #     }
# # # #     .stButton>button {
# # # #         background-color: #FF4B4B;
# # # #         color: white;
# # # #         border: none;
# # # #         padding: 10px 20px;
# # # #         border-radius: 5px;
# # # #         transition: background-color 0.3s;
# # # #     }
# # # #     .stButton>button:hover {
# # # #         background-color: #FF2E2E;
# # # #     }
# # # #     </style>
# # # # """, unsafe_allow_html=True)

# # # # def create_pdf_report(image, results, model):
# # # #     pdf = FPDF()
# # # #     pdf.add_page()
    
# # # #     # Header
# # # #     pdf.set_font('Arial', 'B', 16)
# # # #     pdf.cell(0, 10, 'Traffic Sign Detection Report', 0, 1, 'C')
# # # #     pdf.ln(10)
    
# # # #     # Save detection image
# # # #     cv2.imwrite("temp_detection.jpg", results.plot())
    
# # # #     # Add images
# # # #     pdf.image("temp_detection.jpg", x=10, w=190)
# # # #     pdf.ln(10)
    
# # # #     # Detection details
# # # #     pdf.set_font('Arial', 'B', 14)
# # # #     pdf.cell(0, 10, f'Detected Objects: {len(results.boxes)}', 0, 1)
    
# # # #     pdf.set_font('Arial', '', 12)
# # # #     for idx, box in enumerate(results.boxes):
# # # #         class_id = box.cls.cpu().numpy()[0]
# # # #         conf = box.conf.cpu().numpy()[0]
# # # #         class_name = model.names[int(class_id)]
# # # #         pdf.cell(0, 10, f'Detection {idx+1}: {class_name} (Confidence: {conf:.2%})', 0, 1)
    
# # # #     # Save PDF to memory
# # # #     pdf_output = io.BytesIO()
# # # #     pdf.output(pdf_output)
# # # #     pdf_output.seek(0)
    
# # # #     return pdf_output

# # # # @st.cache_resource
# # # # def load_model():
# # # #     return YOLO("best.pt")

# # # # def create_download_link(pdf_bytes):
# # # #     b64 = base64.b64encode(pdf_bytes.read()).decode()
# # # #     return f'<a href="data:application/pdf;base64,{b64}" download="detection_report.pdf">Download PDF Report</a>'

# # # # def main():
# # # #     col1, col2, col3 = st.columns([1,2,1])
# # # #     with col2:
# # # #         st.title("🚦 Traffic Sign Detection")
# # # #         st.markdown("### Advanced AI-Powered Traffic Sign Recognition")
    
# # # #     with st.spinner("Loading AI Model..."):
# # # #         model = load_model()
    
# # # #     st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
# # # #     uploaded_file = st.file_uploader("Drop your image here or click to upload", type=['jpg', 'jpeg', 'png'])
# # # #     st.markdown("</div>", unsafe_allow_html=True)
    
# # # #     if uploaded_file:
# # # #         image_bytes = uploaded_file.read()
# # # #         image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
# # # #         col1, col2 = st.columns(2)
        
# # # #         with col1:
# # # #             st.markdown("### Original Image")
# # # #             st.image(image, channels="BGR", use_container_width=True)
        
# # # #         if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
# # # #             with st.spinner("Analyzing image..."):
# # # #                 progress_bar = st.progress(0)
# # # #                 for i in range(100):
# # # #                     time.sleep(0.01)
# # # #                     progress_bar.progress(i + 1)
                
# # # #                 image_resized = cv2.resize(image, (640, 640))
# # # #                 results = model.predict(source=image_resized, conf=0.25)[0]
# # # #                 plotted_image = results.plot()
                
# # # #                 with col2:
# # # #                     st.markdown("### Detection Results")
# # # #                     st.image(plotted_image, channels="BGR", use_container_width=True)
                
# # # #                 # Generate and offer PDF download
# # # #                 pdf_output = create_pdf_report(image, results, model)
# # # #                 st.markdown(create_download_link(pdf_output), unsafe_allow_html=True)
                
# # # #                 st.markdown("<div class='detection-box'>", unsafe_allow_html=True)
# # # #                 st.markdown("### 📊 Detailed Analysis")
                
# # # #                 if len(results.boxes) > 0:
# # # #                     for idx, box in enumerate(results.boxes):
# # # #                         class_id = box.cls.cpu().numpy()[0]
# # # #                         conf = box.conf.cpu().numpy()[0]
# # # #                         class_name = model.names[int(class_id)]
# # # #                         with st.expander(f"Detection {idx+1}: {class_name}"):
# # # #                             st.markdown(f"**Confidence Score:** {conf:.2%}")
# # # #                             st.progress(float(conf))
# # # #                             box_coords = box.xyxy.cpu().numpy()[0]
# # # #                             st.code(f"X1: {box_coords[0]:.1f}, Y1: {box_coords[1]:.1f}\nX2: {box_coords[2]:.1f}, Y2: {box_coords[3]:.1f}")
# # # #                 else:
# # # #                     st.warning("No traffic signs detected in this image.")
                
# # # #                 col1, col2, col3 = st.columns(3)
# # # #                 with col1:
# # # #                     st.metric("Total Detections", len(results.boxes))
# # # #                 with col2:
# # # #                     avg_conf = np.mean(results.boxes.conf.cpu().numpy()) if len(results.boxes) > 0 else 0
# # # #                     st.metric("Average Confidence", f"{avg_conf:.2%}")
# # # #                 with col3:
# # # #                     unique_classes = len(set(results.boxes.cls.cpu().numpy())) if len(results.boxes) > 0 else 0
# # # #                     st.metric("Unique Sign Types", unique_classes)

# # # #     st.markdown("---")
# # # #     st.markdown("Made with ❤️ using YOLOv8 and Streamlit")

# # # # if __name__ == "__main__":
# # # #     main()





# # # # import streamlit as st
# # # # from ultralytics import YOLO
# # # # import cv2
# # # # import numpy as np
# # # # from PIL import Image
# # # # import time
# # # # import io
# # # # import base64
# # # # from fpdf import FPDF

# # # # # Page config
# # # # st.set_page_config(page_title="Traffic Sign Detection", page_icon="🚦", layout="wide")

# # # # # Custom CSS
# # # # st.markdown("""
# # # #     <style>
# # # #     .stApp {
# # # #         background: linear-gradient(to right, #1a1a1a, #2d2d2d);
# # # #         color: white;
# # # #     }
# # # #     .upload-box {
# # # #         border: 2px dashed #4a4a4a;
# # # #         border-radius: 10px;
# # # #         padding: 20px;
# # # #         text-align: center;
# # # #         background: rgba(255,255,255,0.05);
# # # #     }
# # # #     .detection-box {
# # # #         background: rgba(255,255,255,0.1);
# # # #         padding: 20px;
# # # #         border-radius: 10px;
# # # #         margin: 10px 0;
# # # #     }
# # # #     .stButton>button {
# # # #         background-color: #FF4B4B !important;
# # # #         color: white !important;
# # # #         border: none !important;
# # # #         padding: 10px 20px !important;
# # # #         border-radius: 5px !important;
# # # #         transition: background-color 0.3s !important;
# # # #     }
# # # #     .stButton>button:hover {
# # # #         background-color: #FF2E2E !important;
# # # #     }
# # # #     .download-button {
# # # #         display: inline-block;
# # # #         padding: 10px 20px;
# # # #         background-color: #4CAF50;
# # # #         color: white;
# # # #         text-decoration: none;
# # # #         border-radius: 5px;
# # # #         margin: 10px 0;
# # # #         transition: background-color 0.3s;
# # # #     }
# # # #     .download-button:hover {
# # # #         background-color: #45a049;
# # # #     }
# # # #     .confidence-bar {
# # # #         background-color: #4CAF50;
# # # #         height: 20px;
# # # #         border-radius: 10px;
# # # #         transition: width 0.3s;
# # # #     }
# # # #     </style>
# # # # """, unsafe_allow_html=True)

# # # # def create_pdf_report(image, results, model):
# # # #     pdf = FPDF()
# # # #     pdf.add_page()
    
# # # #     # Header
# # # #     pdf.set_font('Arial', 'B', 16)
# # # #     pdf.cell(0, 10, 'Traffic Sign Detection Report', 0, 1, 'C')
# # # #     pdf.ln(10)
    
# # # #     # Save detection image
# # # #     is_success, buffer = cv2.imencode(".jpg", results.plot())
# # # #     if is_success:
# # # #         pdf.image(io.BytesIO(buffer), x=10, w=190)
# # # #     pdf.ln(10)
    
# # # #     # Detection details
# # # #     pdf.set_font('Arial', 'B', 14)
# # # #     pdf.cell(0, 10, f'Detected Objects: {len(results.boxes)}', 0, 1)
    
# # # #     # Summary statistics
# # # #     pdf.set_font('Arial', 'B', 12)
# # # #     pdf.cell(0, 10, 'Summary Statistics:', 0, 1)
# # # #     pdf.set_font('Arial', '', 12)
    
# # # #     avg_conf = np.mean(results.boxes.conf.cpu().numpy()) if len(results.boxes) > 0 else 0
# # # #     unique_classes = len(set(results.boxes.cls.cpu().numpy())) if len(results.boxes) > 0 else 0
    
# # # #     pdf.cell(0, 10, f'Average Confidence: {avg_conf:.2%}', 0, 1)
# # # #     pdf.cell(0, 10, f'Unique Sign Types: {unique_classes}', 0, 1)
# # # #     pdf.ln(10)
    
# # # #     # Detailed detections
# # # #     pdf.set_font('Arial', 'B', 12)
# # # #     pdf.cell(0, 10, 'Detailed Detections:', 0, 1)
# # # #     pdf.set_font('Arial', '', 12)
    
# # # #     for idx, box in enumerate(results.boxes):
# # # #         class_id = box.cls.cpu().numpy()[0]
# # # #         conf = box.conf.cpu().numpy()[0]
# # # #         class_name = model.names[int(class_id)]
# # # #         box_coords = box.xyxy.cpu().numpy()[0]
        
# # # #         pdf.cell(0, 10, f'Detection {idx+1}: {class_name}', 0, 1)
# # # #         pdf.cell(0, 10, f'Confidence: {conf:.2%}', 0, 1)
# # # #         pdf.cell(0, 10, f'Location: X1={box_coords[0]:.1f}, Y1={box_coords[1]:.1f}, X2={box_coords[2]:.1f}, Y2={box_coords[3]:.1f}', 0, 1)
# # # #         pdf.ln(5)
    
# # # #     return pdf.output(dest='S').encode('latin1')

# # # # def create_download_link(pdf_bytes):
# # # #     b64 = base64.b64encode(pdf_bytes).decode()
# # # #     return f'<a href="data:application/pdf;base64,{b64}" download="detection_report.pdf" class="download-button">📄 Download PDF Report</a>'

# # # # @st.cache_resource
# # # # def load_model():
# # # #     return YOLO("best.pt")

# # # # def main():
# # # #     # Header
# # # #     col1, col2, col3 = st.columns([1,2,1])
# # # #     with col2:
# # # #         st.title("🚦 Traffic Sign Detection")
# # # #         st.markdown("### Advanced AI-Powered Traffic Sign Recognition")
    
# # # #     # Load model
# # # #     with st.spinner("Loading AI Model..."):
# # # #         model = load_model()
    
# # # #     # Upload section
# # # #     st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
# # # #     uploaded_file = st.file_uploader("Drop your image here or click to upload", type=['jpg', 'jpeg', 'png'])
# # # #     st.markdown("</div>", unsafe_allow_html=True)
    
# # # #     if uploaded_file:
# # # #         # Process image
# # # #         image_bytes = uploaded_file.read()
# # # #         image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
# # # #         # Display columns
# # # #         col1, col2 = st.columns(2)
        
# # # #         with col1:
# # # #             st.markdown("### Original Image")
# # # #             st.image(image, channels="BGR", use_container_width=True)
        
# # # #         # Analyze button
# # # #         if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
# # # #             with st.spinner("Analyzing image..."):
# # # #                 # Progress animation
# # # #                 progress_bar = st.progress(0)
# # # #                 for i in range(100):
# # # #                     time.sleep(0.01)
# # # #                     progress_bar.progress(i + 1)
                
# # # #                 # Run detection
# # # #                 image_resized = cv2.resize(image, (640, 640))
# # # #                 results = model.predict(source=image_resized, conf=0.25)[0]
# # # #                 plotted_image = results.plot()
                
# # # #                 # Show results
# # # #                 with col2:
# # # #                     st.markdown("### Detection Results")
# # # #                     st.image(plotted_image, channels="BGR", use_container_width=True)
                
# # # #                 # Generate and display PDF download
# # # #                 pdf_bytes = create_pdf_report(image, results, model)
# # # #                 st.markdown(create_download_link(pdf_bytes), unsafe_allow_html=True)
                
# # # #                 # Detailed analysis
# # # #                 st.markdown("<div class='detection-box'>", unsafe_allow_html=True)
# # # #                 st.markdown("### 📊 Detailed Analysis")
                
# # # #                 if len(results.boxes) > 0:
# # # #                     for idx, box in enumerate(results.boxes):
# # # #                         class_id = box.cls.cpu().numpy()[0]
# # # #                         conf = box.conf.cpu().numpy()[0]
# # # #                         class_name = model.names[int(class_id)]
                        
# # # #                         with st.expander(f"Detection {idx+1}: {class_name}"):
# # # #                             st.markdown(f"**Confidence Score:** {conf:.2%}")
# # # #                             st.progress(float(conf))
                            
# # # #                             box_coords = box.xyxy.cpu().numpy()[0]
# # # #                             st.code(f"""
# # # # Location Details:
# # # # X1: {box_coords[0]:.1f}, Y1: {box_coords[1]:.1f}
# # # # X2: {box_coords[2]:.1f}, Y2: {box_coords[3]:.1f}
# # # # """)
# # # #                 else:
# # # #                     st.warning("No traffic signs detected in this image.")
                
# # # #                 # Summary metrics
# # # #                 col1, col2, col3 = st.columns(3)
# # # #                 with col1:
# # # #                     st.metric("Total Detections", len(results.boxes))
# # # #                 with col2:
# # # #                     avg_conf = np.mean(results.boxes.conf.cpu().numpy()) if len(results.boxes) > 0 else 0
# # # #                     st.metric("Average Confidence", f"{avg_conf:.2%}")
# # # #                 with col3:
# # # #                     unique_classes = len(set(results.boxes.cls.cpu().numpy())) if len(results.boxes) > 0 else 0
# # # #                     st.metric("Unique Sign Types", unique_classes)

# # # #     # Footer
# # # #     st.markdown("---")
# # # #     st.markdown("Made with ❤️ using YOLOv8 and Streamlit")

# # # # if __name__ == "__main__":
# # # #     main()












# # # import streamlit as st
# # # from ultralytics import YOLO
# # # import cv2
# # # import numpy as np
# # # from PIL import Image
# # # import time

# # # # Page config
# # # st.set_page_config(
# # #     page_title="Traffic Sign Detection",
# # #     page_icon="🚦",
# # #     layout="wide"
# # # )

# # # # Custom CSS
# # # st.markdown("""
# # #     <style>
# # #     .stApp {
# # #         background: linear-gradient(to right, #1a1a1a, #2d2d2d);
# # #         color: white;
# # #     }
# # #     .upload-box {
# # #         border: 2px dashed #4a4a4a;
# # #         border-radius: 10px;
# # #         padding: 20px;
# # #         text-align: center;
# # #         background: rgba(255,255,255,0.05);
# # #     }
# # #     .detection-box {
# # #         background: rgba(255,255,255,0.1);
# # #         padding: 20px;
# # #         border-radius: 10px;
# # #         margin: 10px 0;
# # #     }
# # #     .confidence-meter {
# # #         height: 20px;
# # #         background: green;
# # #         border-radius: 10px;
# # #         overflow: hidden;
# # #     }
# # #     </style>
# # # """, unsafe_allow_html=True)

# # # @st.cache_resource
# # # def load_model():
# # #     return YOLO("best.pt")

# # # def create_confidence_bar(confidence):
# # #     color = "green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"
# # #     return f"""
# # #         <div class="confidence-meter">
# # #             <div style="width:{confidence*100}%; height:100%; background:{color}; transition:width 0.5s;">
# # #             </div>
# # #         </div>
# # #     """

# # # def main():
# # #     # Header
# # #     col1, col2, col3 = st.columns([1,2,1])
# # #     with col2:
# # #         st.title("🚦 Traffic Sign Detection")
# # #         st.markdown("### Advanced AI-Powered Traffic Sign Recognition")
    
# # #     # Model loading with spinner
# # #     with st.spinner("Loading AI Model..."):
# # #         model = load_model()
    
# # #     # File upload section
# # #     st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
# # #     uploaded_file = st.file_uploader(
# # #         "Drop your image here or click to upload",
# # #         type=['jpg', 'jpeg', 'png']
# # #     )
# # #     st.markdown("</div>", unsafe_allow_html=True)
    
# # #     if uploaded_file:
# # #         # Image processing
# # #         image_bytes = uploaded_file.read()
# # #         image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
# # #         # Display columns
# # #         col1, col2 = st.columns(2)
        
# # #         with col1:
# # #             st.markdown("### Original Image")
# # #             st.image(image, channels="BGR", use_container_width=True)
        
# # #         # Process button
# # #         if st.button("🔍 Analyze Image", use_container_width=True):
# # #             with st.spinner("Analyzing image..."):
# # #                 # Add progress bar
# # #                 progress_bar = st.progress(0)
# # #                 for i in range(100):
# # #                     time.sleep(0.01)
# # #                     progress_bar.progress(i + 1)
                
# # #                 # Inference
# # #                 image_resized = cv2.resize(image, (640, 640))
# # #                 results = model.predict(source=image_resized, conf=0.25)[0]
# # #                 plotted_image = results.plot()
                
# # #                 with col2:
# # #                     st.markdown("### Detection Results")
# # #                     st.image(plotted_image, channels="BGR", use_container_width=True)
                
# # #                 # Results section
# # #                 st.markdown("<div class='detection-box'>", unsafe_allow_html=True)
# # #                 st.markdown("### 📊 Detailed Analysis")
                
# # #                 if len(results.boxes) > 0:
# # #                     for idx, box in enumerate(results.boxes):
# # #                         class_id = box.cls.cpu().numpy()[0]
# # #                         conf = box.conf.cpu().numpy()[0]
# # #                         class_name = model.names[int(class_id)]
                        
# # #                         # Create expandable section for each detection
# # #                         with st.expander(f"Detection {idx+1}: {class_name}"):
# # #                             st.markdown(f"**Confidence Score:** {conf:.2%}")
# # #                             st.markdown(create_confidence_bar(conf), unsafe_allow_html=True)
                            
# # #                             # Box coordinates
# # #                             box_coords = box.xyxy.cpu().numpy()[0]
# # #                             st.markdown("**Location Details:**")
# # #                             st.code(f"X1: {box_coords[0]:.1f}, Y1: {box_coords[1]:.1f}\nX2: {box_coords[2]:.1f}, Y2: {box_coords[3]:.1f}")
# # #                 else:
# # #                     st.warning("No traffic signs detected in this image.")
                
# # #                 st.markdown("</div>", unsafe_allow_html=True)
                
# # #                 # Summary metrics
# # #                 col1, col2, col3 = st.columns(3)
# # #                 with col1:
# # #                     st.metric("Total Detections", len(results.boxes))
# # #                 with col2:
# # #                     avg_conf = np.mean(results.boxes.conf.cpu().numpy()) if len(results.boxes) > 0 else 0
# # #                     st.metric("Average Confidence", f"{avg_conf:.2%}")
# # #                 with col3:
# # #                     unique_classes = len(set(results.boxes.cls.cpu().numpy())) if len(results.boxes) > 0 else 0
# # #                     st.metric("Unique Sign Types", unique_classes)

# # #     # Footer
# # #     st.markdown("---")
# # #     st.markdown(
# # #         "Made with ❤️ using YOLOv8 and Streamlit | "
# # #         "[GitHub](https://github.com) | "
# # #         "[Report Issue](https://github.com/issues)"
# # #     )

# # # if __name__ == "__main__":
# # #     main()



# # # import streamlit as st
# # # from ultralytics import YOLO
# # # import cv2
# # # import numpy as np
# # # from PIL import Image
# # # import time

# # # # Page config
# # # st.set_page_config(
# # #     page_title="Traffic Sign Detection",
# # #     page_icon="🚦",
# # #     layout="wide"
# # # )

# # # # Custom CSS
# # # st.markdown("""
# # #     <style>
# # #     .stApp {
# # #         background: linear-gradient(to right, #1a1a1a, #2d2d2d);
# # #         color: white;
# # #     }
# # #     .upload-box {
# # #         border: 2px dashed #4a4a4a;
# # #         border-radius: 10px;
# # #         padding: 20px;
# # #         text-align: center;
# # #         background: rgba(255,255,255,0.05);
# # #     }
# # #     .detection-box {
# # #         background: rgba(255,255,255,0.1);
# # #         padding: 20px;
# # #         border-radius: 10px;
# # #         margin: 10px 0;
# # #     }
# # #     .confidence-meter {
# # #         height: 20px;
# # #         background: green;
# # #         border-radius: 10px;
# # #         overflow: hidden;
# # #     }
# # #     </style>
# # # """, unsafe_allow_html=True)

# # # @st.cache_resource
# # # def load_model():
# # #     return YOLO("best.pt")

# # # def create_confidence_bar(confidence):
# # #     color = "green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"
# # #     return f"""
# # #         <div class="confidence-meter">
# # #             <div style="width:{confidence*100}%; height:100%; background:{color}; transition:width 0.5s;">
# # #             </div>
# # #         </div>
# # #     """

# # # def main():
# # #     # Header
# # #     col1, col2, col3 = st.columns([1,2,1])
# # #     with col2:
# # #         st.title("🚦 Traffic Sign Detection")
# # #         st.markdown("### Advanced AI-Powered Traffic Sign Recognition")
    
# # #     # Model loading with spinner
# # #     with st.spinner("Loading AI Model..."):
# # #         model = load_model()
    
# # #     # File upload section
# # #     st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
# # #     uploaded_file = st.file_uploader(
# # #         "Drop your image here or click to upload",
# # #         type=['jpg', 'jpeg', 'png']
# # #     )
# # #     st.markdown("</div>", unsafe_allow_html=True)
    
# # #     if uploaded_file:
# # #         # Image processing
# # #         image_bytes = uploaded_file.read()
# # #         image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
# # #         # Display columns
# # #         col1, col2 = st.columns(2)
        
# # #         with col1:
# # #             st.markdown("### Original Image")
# # #             # Convert BGR to RGB for display
# # #             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# # #             st.image(image_rgb, width=None)  # width=None will make it responsive
        
# # #         # Process button
# # #         if st.button("🔍 Analyze Image"):  # Removed width parameter
# # #             with st.spinner("Analyzing image..."):
# # #                 # Add progress bar
# # #                 progress_bar = st.progress(0)
# # #                 for i in range(100):
# # #                     time.sleep(0.01)
# # #                     progress_bar.progress(i + 1)
                
# # #                 # Inference
# # #                 image_resized = cv2.resize(image, (640, 640))
# # #                 results = model.predict(source=image_resized, conf=0.25)[0]
# # #                 plotted_image = results.plot()
                
# # #                 with col2:
# # #                     st.markdown("### Detection Results")
# # #                     # Convert BGR to RGB for display
# # #                     plotted_image_rgb = cv2.cvtColor(plotted_image, cv2.COLOR_BGR2RGB)
# # #                     st.image(plotted_image_rgb, width=None)  # width=None will make it responsive
                
# # #                 # Results section
# # #                 st.markdown("<div class='detection-box'>", unsafe_allow_html=True)
# # #                 st.markdown("### 📊 Detailed Analysis")
                
# # #                 if len(results.boxes) > 0:
# # #                     for idx, box in enumerate(results.boxes):
# # #                         class_id = box.cls.cpu().numpy()[0]
# # #                         conf = box.conf.cpu().numpy()[0]
# # #                         class_name = model.names[int(class_id)]
                        
# # #                         # Create expandable section for each detection
# # #                         with st.expander(f"Detection {idx+1}: {class_name}"):
# # #                             st.markdown(f"**Confidence Score:** {conf:.2%}")
# # #                             st.markdown(create_confidence_bar(conf), unsafe_allow_html=True)
                            
# # #                             # Box coordinates
# # #                             box_coords = box.xyxy.cpu().numpy()[0]
# # #                             st.markdown("**Location Details:**")
# # #                             st.code(f"X1: {box_coords[0]:.1f}, Y1: {box_coords[1]:.1f}\nX2: {box_coords[2]:.1f}, Y2: {box_coords[3]:.1f}")
# # #                 else:
# # #                     st.warning("No traffic signs detected in this image.")
                
# # #                 st.markdown("</div>", unsafe_allow_html=True)
                
# # #                 # Summary metrics
# # #                 col1, col2, col3 = st.columns(3)
# # #                 with col1:
# # #                     st.metric("Total Detections", len(results.boxes))
# # #                 with col2:
# # #                     avg_conf = np.mean(results.boxes.conf.cpu().numpy()) if len(results.boxes) > 0 else 0
# # #                     st.metric("Average Confidence", f"{avg_conf:.2%}")
# # #                 with col3:
# # #                     unique_classes = len(set(results.boxes.cls.cpu().numpy())) if len(results.boxes) > 0 else 0
# # #                     st.metric("Unique Sign Types", unique_classes)

# # #     # Footer
# # #     st.markdown("---")
# # #     st.markdown(
# # #         "Made with ❤️ using YOLOv8 and Streamlit | "
# # #         "[GitHub](https://github.com) | "
# # #         "[Report Issue](https://github.com/issues)"
# # #     )

# # # if __name__ == "__main__":
# # #     main()
























# # # import streamlit as st
# # # from ultralytics import YOLO
# # # import cv2
# # # import numpy as np
# # # from PIL import Image
# # # import time
# # # import pandas as pd
# # # from datetime import datetime
# # # import plotly.express as px
# # # import plotly.graph_objects as go
# # # import io
# # # import base64
# # # from fpdf import FPDF
# # # import json
# # # import os

# # # # Create directory for saving analysis if it doesn't exist
# # # if not os.path.exists("analysis_history"):
# # #     os.makedirs("analysis_history")

# # # # Page config
# # # st.set_page_config(
# # #     page_title="Advanced Traffic Sign Detection System",
# # #     page_icon="🚦",
# # #     layout="wide",
# # #     initial_sidebar_state="expanded"
# # # )

# # # # Custom CSS with improved styling
# # # st.markdown("""
# # #     <style>
# # #     .stApp {
# # #         background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
# # #         color: #e5e5e5;
# # #     }
# # #     .upload-box {
# # #         border: 2px dashed #4a4a8a;
# # #         border-radius: 15px;
# # #         padding: 30px;
# # #         text-align: center;
# # #         background: rgba(255,255,255,0.05);
# # #         backdrop-filter: blur(10px);
# # #         transition: all 0.3s ease;
# # #     }
# # #     .upload-box:hover {
# # #         border-color: #6a6aaa;
# # #         background: rgba(255,255,255,0.08);
# # #     }
# # #     .detection-box {
# # #         background: rgba(255,255,255,0.07);
# # #         padding: 25px;
# # #         border-radius: 15px;
# # #         margin: 15px 0;
# # #         box-shadow: 0 4px 15px rgba(0,0,0,0.2);
# # #     }
# # #     .metrics-card {
# # #         background: rgba(255,255,255,0.1);
# # #         padding: 20px;
# # #         border-radius: 10px;
# # #         margin: 10px 0;
# # #         transition: transform 0.3s ease;
# # #     }
# # #     .metrics-card:hover {
# # #         transform: translateY(-5px);
# # #     }
# # #     .confidence-meter {
# # #         height: 20px;
# # #         background: rgba(255,255,255,0.1);
# # #         border-radius: 10px;
# # #         overflow: hidden;
# # #         box-shadow: inset 0 0 5px rgba(0,0,0,0.2);
# # #     }
# # #     .header-container {
# # #         padding: 20px;
# # #         background: rgba(255,255,255,0.05);
# # #         border-radius: 15px;
# # #         margin-bottom: 20px;
# # #         text-align: center;
# # #     }
# # #     .stButton>button {
# # #         background: linear-gradient(45deg, #3498db, #2980b9);
# # #         color: white;
# # #         border: none;
# # #         padding: 10px 20px;
# # #         border-radius: 8px;
# # #         transition: all 0.3s ease;
# # #     }
# # #     .stButton>button:hover {
# # #         transform: translateY(-2px);
# # #         box-shadow: 0 5px 15px rgba(0,0,0,0.2);
# # #     }
# # #     </style>
# # # """, unsafe_allow_html=True)

# # # # Caching the model loading
# # # @st.cache_resource
# # # def load_model():
# # #     return YOLO("best.pt")

# # # def create_confidence_bar(confidence):
# # #     colors = {
# # #         "high": "#00ff00",
# # #         "medium": "#ffaa00",
# # #         "low": "#ff0000"
# # #     }
# # #     color = colors["high"] if confidence > 0.7 else colors["medium"] if confidence > 0.5 else colors["low"]
# # #     return f"""
# # #         <div class="confidence-meter">
# # #             <div style="width:{confidence*100}%; height:100%; background:{color}; 
# # #                  transition:width 0.5s; text-align:center; color:white; line-height:20px;">
# # #                 {confidence:.1%}
# # #             </div>
# # #         </div>
# # #     """

# # # def generate_pdf_report(results, image_path, analysis_data):
# # #     pdf = FPDF()
# # #     pdf.add_page()
    
# # #     # Header
# # #     pdf.set_font('Arial', 'B', 16)
# # #     pdf.cell(0, 10, 'Traffic Sign Detection Analysis Report', 0, 1, 'C')
# # #     pdf.set_font('Arial', '', 12)
# # #     pdf.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
# # #     pdf.cell(0, 10, f'Analyzed by: Saira Khan', 0, 1, 'C')
# # #     pdf.ln(10)
    
# # #     # Analysis Summary
# # #     pdf.set_font('Arial', 'B', 14)
# # #     pdf.cell(0, 10, 'Analysis Summary', 0, 1, 'L')
# # #     pdf.set_font('Arial', '', 12)
# # #     pdf.cell(0, 10, f'Total Detections: {len(results.boxes)}', 0, 1, 'L')
    
# # #     if len(results.boxes) > 0:
# # #         avg_conf = np.mean(results.boxes.conf.cpu().numpy())
# # #         pdf.cell(0, 10, f'Average Confidence: {avg_conf:.2%}', 0, 1, 'L')
        
# # #         # Detailed Detections
# # #         pdf.ln(10)
# # #         pdf.set_font('Arial', 'B', 14)
# # #         pdf.cell(0, 10, 'Detailed Detections', 0, 1, 'L')
        
# # #         for idx, box in enumerate(results.boxes):
# # #             pdf.set_font('Arial', 'B', 12)
# # #             class_id = box.cls.cpu().numpy()[0]
# # #             conf = box.conf.cpu().numpy()[0]
# # #             class_name = model.names[int(class_id)]
# # #             pdf.cell(0, 10, f'Detection {idx+1}: {class_name}', 0, 1, 'L')
# # #             pdf.set_font('Arial', '', 12)
# # #             pdf.cell(0, 10, f'Confidence: {conf:.2%}', 0, 1, 'L')
    
# # #     # Save PDF
# # #     report_path = f"analysis_history/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
# # #     pdf.output(report_path)
# # #     return report_path

# # # def create_visualization(results):
# # #     if len(results.boxes) > 0:
# # #         # Prepare data for visualization
# # #         confidences = results.boxes.conf.cpu().numpy()
# # #         classes = [model.names[int(cls)] for cls in results.boxes.cls.cpu().numpy()]
        
# # #         # Create confidence distribution plot
# # #         fig_conf = go.Figure(data=[go.Histogram(x=confidences, nbinsx=20)])
# # #         fig_conf.update_layout(
# # #             title="Confidence Score Distribution",
# # #             xaxis_title="Confidence Score",
# # #             yaxis_title="Count",
# # #             template="plotly_dark"
# # #         )
        
# # #         # Create class distribution plot
# # #         class_counts = pd.Series(classes).value_counts()
# # #         fig_class = px.pie(values=class_counts.values, names=class_counts.index,
# # #                           title="Distribution of Detected Signs")
# # #         fig_class.update_layout(template="plotly_dark")
        
# # #         return fig_conf, fig_class
# # #     return None, None

# # # def main():
# # #     # Header
# # #     st.markdown("""
# # #         <div class="header-container">
# # #             <h1>🚦 Advanced Traffic Sign Detection System</h1>
# # #             <p>Final Year Project by Saira Khan</p>
# # #             <p>A state-of-the-art computer vision system for detecting and analyzing traffic signs</p>
# # #         </div>
# # #     """, unsafe_allow_html=True)
    
# # #     # Sidebar
# # #     with st.sidebar:
# # #         st.header("📊 Analysis Settings")
# # #         confidence_threshold = st.slider("Detection Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
# # #         enable_analytics = st.checkbox("Enable Advanced Analytics", True)
# # #         save_report = st.checkbox("Generate PDF Report", True)
    
# # #     # Model loading with spinner
# # #     with st.spinner("Initializing AI Model..."):
# # #         model = load_model()
    
# # #     # Main content
# # #     col1, col2 = st.columns([2, 1])
    
# # #     with col1:
# # #         st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
# # #         uploaded_file = st.file_uploader(
# # #             "Drop your traffic sign image here or click to upload",
# # #             type=['jpg', 'jpeg', 'png']
# # #         )
# # #         st.markdown("</div>", unsafe_allow_html=True)
    
# # #     with col2:
# # #         st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
# # #         st.markdown("### 📈 System Statistics")
# # #         if 'total_analyses' not in st.session_state:
# # #             st.session_state.total_analyses = 0
# # #         st.metric("Total Analyses", st.session_state.total_analyses)
# # #         st.markdown("</div>", unsafe_allow_html=True)
    
# # #     if uploaded_file:
# # #         image_bytes = uploaded_file.read()
# # #         image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
# # #         col1, col2 = st.columns(2)
        
# # #         with col1:
# # #             st.markdown("### Original Image")
# # #             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# # #             st.image(image_rgb, width=None)
        
# # #         if st.button("🔍 Analyze Image", key="analyze_button"):
# # #             st.session_state.total_analyses += 1
            
# # #             with st.spinner("Processing image..."):
# # #                 # Progress bar with custom message
# # #                 progress_text = "Running advanced analysis..."
# # #                 progress_bar = st.progress(0)
# # #                 for i in range(100):
# # #                     progress_bar.progress(i + 1)
# # #                     if i == 30:
# # #                         progress_text = "Detecting traffic signs..."
# # #                     elif i == 60:
# # #                         progress_text = "Analyzing patterns..."
# # #                     elif i == 90:
# # #                         progress_text = "Preparing results..."
# # #                     time.sleep(0.01)
# # #                     st.write(progress_text)
                
# # #                 # Inference
# # #                 image_resized = cv2.resize(image, (640, 640))
# # #                 results = model.predict(source=image_resized, conf=confidence_threshold)[0]
# # #                 plotted_image = results.plot()
                
# # #                 with col2:
# # #                     st.markdown("### Detection Results")
# # #                     plotted_image_rgb = cv2.cvtColor(plotted_image, cv2.COLOR_BGR2RGB)
# # #                     st.image(plotted_image_rgb, width=None)
                
# # #                 # Advanced Analytics
# # #                 if enable_analytics:
# # #                     st.markdown("### 📊 Advanced Analytics")
# # #                     fig_conf, fig_class = create_visualization(results)
# # #                     if fig_conf and fig_class:
# # #                         col1, col2 = st.columns(2)
# # #                         with col1:
# # #                             st.plotly_chart(fig_conf, use_container_width=True)
# # #                         with col2:
# # #                             st.plotly_chart(fig_class, use_container_width=True)
                
# # #                 # Detailed Analysis
# # #                 st.markdown("<div class='detection-box'>", unsafe_allow_html=True)
# # #                 st.markdown("### 🎯 Detection Details")
                
# # #                 if len(results.boxes) > 0:
# # #                     for idx, box in enumerate(results.boxes):
# # #                         class_id = box.cls.cpu().numpy()[0]
# # #                         conf = box.conf.cpu().numpy()[0]
# # #                         class_name = model.names[int(class_id)]
                        
# # #                         with st.expander(f"Detection {idx+1}: {class_name} ({conf:.2%})"):
# # #                             st.markdown(create_confidence_bar(conf), unsafe_allow_html=True)
                            
# # #                             box_coords = box.xyxy.cpu().numpy()[0]
# # #                             col1, col2 = st.columns(2)
# # #                             with col1:
# # #                                 st.markdown("**Location Coordinates:**")
# # #                                 st.code(f"X1: {box_coords[0]:.1f}, Y1: {box_coords[1]:.1f}\nX2: {box_coords[2]:.1f}, Y2: {box_coords[3]:.1f}")
# # #                             with col2:
# # #                                 st.markdown("**Detection Details:**")
# # #                                 st.markdown(f"• Sign Type: {class_name}")
# # #                                 st.markdown(f"• Confidence: {conf:.2%}")
# # #                 else:
# # #                     st.warning("No traffic signs detected in this image.")
                
# # #                 # Summary metrics with improved styling
# # #                 col1, col2, col3 = st.columns(3)
# # #                 with col1:
# # #                     st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
# # #                     st.metric("Total Detections", len(results.boxes))
# # #                     st.markdown("</div>", unsafe_allow_html=True)
# # #                 with col2:
# # #                     st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
# # #                     avg_conf = np.mean(results.boxes.conf.cpu().numpy()) if len(results.boxes) > 0 else 0
# # #                     st.metric("Average Confidence", f"{avg_conf:.2%}")
# # #                     st.markdown("</div>", unsafe_allow_html=True)
# # #                 with col3:
# # #                     st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
# # #                     unique_classes = len(set(results.boxes.cls.cpu().numpy())) if len(results.boxes) > 0 else 0
# # #                     st.metric("Unique Sign Types", unique_classes)
# # #                     st.markdown("</div>", unsafe_allow_html=True)
                
# # #                 # Generate and save report
# # #                 if save_report:
# # #                     report_path = generate_pdf_report(results, uploaded_file.name, {
# # #                         'total_detections': len(results.boxes),
# # #                         'average_confidence': avg_conf if len(results.boxes) > 0 else 0,
# # #                         'unique_signs': unique_classes
# # #                     })
                    
# # #                     with open(report_path, "rb") as file:
# # #                         btn = st.download_button(
# # #                             label="📄 Download Analysis Report",
# # #                             data=file,
# # #                             file_name=f"traffic_sign_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
# # #                             mime="application/pdf"
# # #                         )

# # #     # Footer
# # #     st.markdown("---")
# # #     st.markdown(
# # #         """
# # #         <div style='text-align: center'>
# # #             <p>Advanced Traffic Sign Detection System | Final Year Project</p>
# # #             <p>Developed by Saira Khan | 2024</p>
# # #             <p>Using YOLOv8 and Streamlit</p>
# # #         </div>
# # #         """, 
# # #         unsafe_allow_html=True
# # #     )

# # # if __name__ == "__main__":
# # #     main()







# # # import streamlit as st
# # # from ultralytics import YOLO
# # # import cv2
# # # import numpy as np
# # # from PIL import Image
# # # import time
# # # from fpdf import FPDF
# # # import io
# # # import base64

# # # st.set_page_config(page_title="Traffic Sign Detection", page_icon="🚦", layout="wide")

# # # st.markdown("""
# # #     <style>
# # #     .stApp {
# # #         background: linear-gradient(to right, #1a1a1a, #2d2d2d);
# # #         color: white;
# # #     }
# # #     .upload-box {
# # #         border: 2px dashed #4a4a4a;
# # #         border-radius: 10px;
# # #         padding: 20px;
# # #         text-align: center;
# # #         background: rgba(255,255,255,0.05);
# # #     }
# # #     .detection-box {
# # #         background: rgba(255,255,255,0.1);
# # #         padding: 20px;
# # #         border-radius: 10px;
# # #         margin: 10px 0;
# # #     }
# # #     .stButton>button {
# # #         background-color: #FF4B4B;
# # #         color: white;
# # #         border: none;
# # #         padding: 10px 20px;
# # #         border-radius: 5px;
# # #         transition: background-color 0.3s;
# # #     }
# # #     .stButton>button:hover {
# # #         background-color: #FF2E2E;
# # #     }
# # #     </style>
# # # """, unsafe_allow_html=True)

# # # def create_pdf_report(image, results, model):
# # #     pdf = FPDF()
# # #     pdf.add_page()
    
# # #     # Header
# # #     pdf.set_font('Arial', 'B', 16)
# # #     pdf.cell(0, 10, 'Traffic Sign Detection Report', 0, 1, 'C')
# # #     pdf.ln(10)
    
# # #     # Save detection image
# # #     cv2.imwrite("temp_detection.jpg", results.plot())
    
# # #     # Add images
# # #     pdf.image("temp_detection.jpg", x=10, w=190)
# # #     pdf.ln(10)
    
# # #     # Detection details
# # #     pdf.set_font('Arial', 'B', 14)
# # #     pdf.cell(0, 10, f'Detected Objects: {len(results.boxes)}', 0, 1)
    
# # #     pdf.set_font('Arial', '', 12)
# # #     for idx, box in enumerate(results.boxes):
# # #         class_id = box.cls.cpu().numpy()[0]
# # #         conf = box.conf.cpu().numpy()[0]
# # #         class_name = model.names[int(class_id)]
# # #         pdf.cell(0, 10, f'Detection {idx+1}: {class_name} (Confidence: {conf:.2%})', 0, 1)
    
# # #     # Save PDF to memory
# # #     pdf_output = io.BytesIO()
# # #     pdf.output(pdf_output)
# # #     pdf_output.seek(0)
    
# # #     return pdf_output

# # # @st.cache_resource
# # # def load_model():
# # #     return YOLO("best.pt")

# # # def create_download_link(pdf_bytes):
# # #     b64 = base64.b64encode(pdf_bytes.read()).decode()
# # #     return f'<a href="data:application/pdf;base64,{b64}" download="detection_report.pdf">Download PDF Report</a>'

# # # def main():
# # #     col1, col2, col3 = st.columns([1,2,1])
# # #     with col2:
# # #         st.title("🚦 Traffic Sign Detection")
# # #         st.markdown("### Advanced AI-Powered Traffic Sign Recognition")
    
# # #     with st.spinner("Loading AI Model..."):
# # #         model = load_model()
    
# # #     st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
# # #     uploaded_file = st.file_uploader("Drop your image here or click to upload", type=['jpg', 'jpeg', 'png'])
# # #     st.markdown("</div>", unsafe_allow_html=True)
    
# # #     if uploaded_file:
# # #         image_bytes = uploaded_file.read()
# # #         image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
# # #         col1, col2 = st.columns(2)
        
# # #         with col1:
# # #             st.markdown("### Original Image")
# # #             st.image(image, channels="BGR", use_container_width=True)
        
# # #         if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
# # #             with st.spinner("Analyzing image..."):
# # #                 progress_bar = st.progress(0)
# # #                 for i in range(100):
# # #                     time.sleep(0.01)
# # #                     progress_bar.progress(i + 1)
                
# # #                 image_resized = cv2.resize(image, (640, 640))
# # #                 results = model.predict(source=image_resized, conf=0.25)[0]
# # #                 plotted_image = results.plot()
                
# # #                 with col2:
# # #                     st.markdown("### Detection Results")
# # #                     st.image(plotted_image, channels="BGR", use_container_width=True)
                
# # #                 # Generate and offer PDF download
# # #                 pdf_output = create_pdf_report(image, results, model)
# # #                 st.markdown(create_download_link(pdf_output), unsafe_allow_html=True)
                
# # #                 st.markdown("<div class='detection-box'>", unsafe_allow_html=True)
# # #                 st.markdown("### 📊 Detailed Analysis")
                
# # #                 if len(results.boxes) > 0:
# # #                     for idx, box in enumerate(results.boxes):
# # #                         class_id = box.cls.cpu().numpy()[0]
# # #                         conf = box.conf.cpu().numpy()[0]
# # #                         class_name = model.names[int(class_id)]
# # #                         with st.expander(f"Detection {idx+1}: {class_name}"):
# # #                             st.markdown(f"**Confidence Score:** {conf:.2%}")
# # #                             st.progress(float(conf))
# # #                             box_coords = box.xyxy.cpu().numpy()[0]
# # #                             st.code(f"X1: {box_coords[0]:.1f}, Y1: {box_coords[1]:.1f}\nX2: {box_coords[2]:.1f}, Y2: {box_coords[3]:.1f}")
# # #                 else:
# # #                     st.warning("No traffic signs detected in this image.")
                
# # #                 col1, col2, col3 = st.columns(3)
# # #                 with col1:
# # #                     st.metric("Total Detections", len(results.boxes))
# # #                 with col2:
# # #                     avg_conf = np.mean(results.boxes.conf.cpu().numpy()) if len(results.boxes) > 0 else 0
# # #                     st.metric("Average Confidence", f"{avg_conf:.2%}")
# # #                 with col3:
# # #                     unique_classes = len(set(results.boxes.cls.cpu().numpy())) if len(results.boxes) > 0 else 0
# # #                     st.metric("Unique Sign Types", unique_classes)

# # #     st.markdown("---")
# # #     st.markdown("Made with ❤️ using YOLOv8 and Streamlit")

# # # if __name__ == "__main__":
# # #     main()









# # import streamlit as st
# # from ultralytics import YOLO
# # import cv2
# # import numpy as np
# # from PIL import Image
# # import time
# # import pandas as pd
# # from datetime import datetime
# # import plotly.express as px
# # import plotly.graph_objects as go
# # import io
# # import base64
# # from fpdf import FPDF
# # import json
# # import os

# # # Create directory for saving analysis if it doesn't exist
# # if not os.path.exists("analysis_history"):
# #     os.makedirs("analysis_history")

# # # Page config
# # st.set_page_config(
# #     page_title="Advanced Traffic Sign Detection System",
# #     page_icon="🚦",
# #     layout="wide",
# #     initial_sidebar_state="expanded"
# # )

# # # Custom CSS with improved styling
# # st.markdown("""
# #     <style>
# #     .stApp {
# #         background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
# #         color: #e5e5e5;
# #     }
# #     .upload-box {
# #         border: 2px dashed #4a4a8a;
# #         border-radius: 15px;
# #         padding: 30px;
# #         text-align: center;
# #         background: rgba(255,255,255,0.05);
# #         backdrop-filter: blur(10px);
# #         transition: all 0.3s ease;
# #     }
# #     .upload-box:hover {
# #         border-color: #6a6aaa;
# #         background: rgba(255,255,255,0.08);
# #     }
# #     .detection-box {
# #         background: rgba(255,255,255,0.07);
# #         padding: 25px;
# #         border-radius: 15px;
# #         margin: 15px 0;
# #         box-shadow: 0 4px 15px rgba(0,0,0,0.2);
# #     }
# #     .metrics-card {
# #         background: rgba(255,255,255,0.1);
# #         padding: 20px;
# #         border-radius: 10px;
# #         margin: 10px 0;
# #         transition: transform 0.3s ease;
# #     }
# #     .metrics-card:hover {
# #         transform: translateY(-5px);
# #     }
# #     .confidence-meter {
# #         height: 20px;
# #         background: rgba(255,255,255,0.1);
# #         border-radius: 10px;
# #         overflow: hidden;
# #         box-shadow: inset 0 0 5px rgba(0,0,0,0.2);
# #     }
# #     .header-container {
# #         padding: 20px;
# #         background: rgba(255,255,255,0.05);
# #         border-radius: 15px;
# #         margin-bottom: 20px;
# #         text-align: center;
# #     }
# #     .stButton>button {
# #         background: linear-gradient(45deg, #3498db, #2980b9);
# #         color: white;
# #         border: none;
# #         padding: 10px 20px;
# #         border-radius: 8px;
# #         transition: all 0.3s ease;
# #     }
# #     .stButton>button:hover {
# #         transform: translateY(-2px);
# #         box-shadow: 0 5px 15px rgba(0,0,0,0.2);
# #     }
# #     </style>
# # """, unsafe_allow_html=True)

# # # Initialize session state for storing the model
# # if 'model' not in st.session_state:
# #     st.session_state.model = None

# # @st.cache_resource
# # def load_model():
# #     """Load and cache the YOLO model"""
# #     return YOLO("best.pt")

# # def create_confidence_bar(confidence):
# #     """Create a visual confidence meter"""
# #     colors = {
# #         "high": "#00ff00",
# #         "medium": "#ffaa00",
# #         "low": "#ff0000"
# #     }
# #     color = colors["high"] if confidence > 0.7 else colors["medium"] if confidence > 0.5 else colors["low"]
# #     return f"""
# #         <div class="confidence-meter">
# #             <div style="width:{confidence*100}%; height:100%; background:{color}; 
# #                  transition:width 0.5s; text-align:center; color:white; line-height:20px;">
# #                 {confidence:.1%}
# #             </div>
# #         </div>
# #     """

# # def create_visualization(results, model):
# #     """Create visualization plots for detection results"""
# #     if len(results.boxes) > 0:
# #         # Prepare data for visualization
# #         confidences = results.boxes.conf.cpu().numpy()
# #         classes = [model.names[int(cls)] for cls in results.boxes.cls.cpu().numpy()]
        
# #         # Create confidence distribution plot
# #         fig_conf = go.Figure(data=[go.Histogram(x=confidences, nbinsx=20)])
# #         fig_conf.update_layout(
# #             title="Confidence Score Distribution",
# #             xaxis_title="Confidence Score",
# #             yaxis_title="Count",
# #             template="plotly_dark"
# #         )
        
# #         # Create class distribution plot
# #         class_counts = pd.Series(classes).value_counts()
# #         fig_class = px.pie(values=class_counts.values, names=class_counts.index,
# #                           title="Distribution of Detected Signs")
# #         fig_class.update_layout(template="plotly_dark")
        
# #         return fig_conf, fig_class
# #     return None, None

# # def generate_pdf_report(results, image_path, analysis_data, model):
# #     """Generate a PDF report of the analysis"""
# #     pdf = FPDF()
# #     pdf.add_page()
    
# #     # Header
# #     pdf.set_font('Arial', 'B', 16)
# #     pdf.cell(0, 10, 'Traffic Sign Detection Analysis Report', 0, 1, 'C')
# #     pdf.set_font('Arial', '', 12)
# #     pdf.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
# #     pdf.cell(0, 10, f'Analyzed by: Saira Khan', 0, 1, 'C')
# #     pdf.ln(10)
    
# #     # Analysis Summary
# #     pdf.set_font('Arial', 'B', 14)
# #     pdf.cell(0, 10, 'Analysis Summary', 0, 1, 'L')
# #     pdf.set_font('Arial', '', 12)
# #     pdf.cell(0, 10, f'Total Detections: {len(results.boxes)}', 0, 1, 'L')
    
# #     if len(results.boxes) > 0:
# #         avg_conf = np.mean(results.boxes.conf.cpu().numpy())
# #         pdf.cell(0, 10, f'Average Confidence: {avg_conf:.2%}', 0, 1, 'L')
        
# #         # Detailed Detections
# #         pdf.ln(10)
# #         pdf.set_font('Arial', 'B', 14)
# #         pdf.cell(0, 10, 'Detailed Detections', 0, 1, 'L')
        
# #         for idx, box in enumerate(results.boxes):
# #             pdf.set_font('Arial', 'B', 12)
# #             class_id = box.cls.cpu().numpy()[0]
# #             conf = box.conf.cpu().numpy()[0]
# #             class_name = model.names[int(class_id)]
# #             pdf.cell(0, 10, f'Detection {idx+1}: {class_name}', 0, 1, 'L')
# #             pdf.set_font('Arial', '', 12)
# #             pdf.cell(0, 10, f'Confidence: {conf:.2%}', 0, 1, 'L')
    
# #     # Save PDF
# #     report_path = f"analysis_history/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
# #     pdf.output(report_path)
# #     return report_path

# # def main():
# #     # Load model at startup
# #     if st.session_state.model is None:
# #         with st.spinner("Loading AI Model..."):
# #             st.session_state.model = load_model()
    
# #     # Header
# #     st.markdown("""
# #         <div class="header-container">
# #             <h1>🚦 Advanced Traffic Sign Detection System</h1>
# #             <p>Final Year Project by Saira Khan</p>
# #             <p>A state-of-the-art computer vision system for detecting and analyzing traffic signs</p>
# #         </div>
# #     """, unsafe_allow_html=True)
    
# #     # Sidebar
# #     with st.sidebar:
# #         st.header("📊 Analysis Settings")
# #         confidence_threshold = st.slider("Detection Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
# #         enable_analytics = st.checkbox("Enable Advanced Analytics", True)
# #         save_report = st.checkbox("Generate PDF Report", True)
    
# #     # Main content
# #     col1, col2 = st.columns([2, 1])
    
# #     with col1:
# #         st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
# #         uploaded_file = st.file_uploader(
# #             "Drop your traffic sign image here or click to upload",
# #             type=['jpg', 'jpeg', 'png']
# #         )
# #         st.markdown("</div>", unsafe_allow_html=True)
    
# #     with col2:
# #         st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
# #         st.markdown("### 📈 System Statistics")
# #         if 'total_analyses' not in st.session_state:
# #             st.session_state.total_analyses = 0
# #         st.metric("Total Analyses", st.session_state.total_analyses)
# #         st.markdown("</div>", unsafe_allow_html=True)
    
# #     if uploaded_file:
# #         image_bytes = uploaded_file.read()
# #         image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
# #         col1, col2 = st.columns(2)
        
# #         with col1:
# #             st.markdown("### Original Image")
# #             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# #             st.image(image_rgb)
        
# #         if st.button("🔍 Analyze Image", key="analyze_button"):
# #             st.session_state.total_analyses += 1
            
# #             with st.spinner("Processing image..."):
# #                 # Progress bar with custom message
# #                 progress_text = "Running advanced analysis..."
# #                 progress_bar = st.progress(0)
# #                 for i in range(100):
# #                     progress_bar.progress(i + 1)
# #                     if i == 30:
# #                         progress_text = "Detecting traffic signs..."
# #                     elif i == 60:
# #                         progress_text = "Analyzing patterns..."
# #                     elif i == 90:
# #                         progress_text = "Preparing results..."
# #                     time.sleep(0.01)
# #                     st.write(progress_text)
                
# #                 # Inference
# #                 image_resized = cv2.resize(image, (640, 640))
# #                 results = st.session_state.model.predict(source=image_resized, conf=confidence_threshold)[0]
# #                 plotted_image = results.plot()
                
# #                 with col2:
# #                     st.markdown("### Detection Results")
# #                     plotted_image_rgb = cv2.cvtColor(plotted_image, cv2.COLOR_BGR2RGB)
# #                     st.image(plotted_image_rgb)
                
# #                 # Advanced Analytics
# #                 if enable_analytics:
# #                     st.markdown("### 📊 Advanced Analytics")
# #                     fig_conf, fig_class = create_visualization(results, st.session_state.model)
# #                     if fig_conf and fig_class:
# #                         col1, col2 = st.columns(2)
# #                         with col1:
# #                             st.plotly_chart(fig_conf, use_container_width=True)
# #                         with col2:
# #                             st.plotly_chart(fig_class, use_container_width=True)
                
# #                 # Detailed Analysis
# #                 st.markdown("<div class='detection-box'>", unsafe_allow_html=True)
# #                 st.markdown("### 🎯 Detection Details")
                
# #                 if len(results.boxes) > 0:
# #                     for idx, box in enumerate(results.boxes):
# #                         class_id = box.cls.cpu().numpy()[0]
# #                         conf = box.conf.cpu().numpy()[0]
# #                         class_name = st.session_state.model.names[int(class_id)]
                        
# #                         with st.expander(f"Detection {idx+1}: {class_name} ({conf:.2%})"):
# #                             st.markdown(create_confidence_bar(conf), unsafe_allow_html=True)
                            
# #                             box_coords = box.xyxy.cpu().numpy()[0]
# #                             col1, col2 = st.columns(2)
# #                             with col1:
# #                                 st.markdown("**Location Coordinates:**")
# #                                 st.code(f"X1: {box_coords[0]:.1f}, Y1: {box_coords[1]:.1f}\nX2: {box_coords[2]:.1f}, Y2: {box_coords[3]:.1f}")
# #                             with col2:
# #                                 st.markdown("**Detection Details:**")
# #                                 st.markdown(f"• Sign Type: {class_name}")
# #                                 st.markdown(f"• Confidence: {conf:.2%}")
# #                 else:
# #                     st.warning("No traffic signs detected in this image.")
                
# #                 # Summary metrics with improved styling
# #                 col1, col2, col3 = st.columns(3)
# #                 with col1:
# #                     st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
# #                     st.metric("Total Detections", len(results.boxes))
# #                     st.markdown("</div>", unsafe_allow_html=True)
# #                 with col2:
# #                     st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
# #                     avg_conf = np.mean(results.boxes.conf.cpu().numpy()) if len(results.boxes) > 0 else 0
# #                     st.metric("Average Confidence", f"{avg_conf:.2%}")
# #                     st.markdown("</div>", unsafe_allow_html=True)
# #                 with col3:
# #                     st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
# #                     unique_classes = len(set(results.boxes.cls.cpu().numpy())) if len(results.boxes) > 0 else 0
# #                     st.metric("Unique Sign Types", unique_classes)
# #                     st.markdown("</div>", unsafe_allow_html=True)
                
# #                 # Generate and save report
# #                 if save_report:
# #                     report_path = generate_pdf_report(results, uploaded_file.name, {
# #                         'total_detections': len(results.boxes),
# #                         'average_confidence': avg_conf if len(results.boxes) > 0 else 0,
# #                         'unique_signs': unique_classes
# #                     }, st.session_state.model)
                    
# #                     with open(report_path, "rb") as file:
# #                         btn = st.download_button(
# #                             label="📄 Download Analysis Report",
# #                             data=file,
# #                             file_name=f"traffic_sign_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
# #                             mime="application/pdf"
# #                         )

# #     # Footer
# #     st.markdown("---")
# #     st.markdown(
# #         """
# #         <div style='text-align: center'>
# #             <p>Advanced Traffic Sign Detection System | Final Year Project</p>
# #             <p>Developed by Saira Khan | 2024</p>
# #             <p>Using YOLOv8 and Streamlit | All Rights Reserved</p>
# #         </div>
# #         """, 
# #         unsafe_allow_html=True
# #     )

# #     # Add help section in the sidebar
# #     with st.sidebar:
# #         st.markdown("---")
# #         with st.expander("ℹ️ Help & Instructions"):
# #             st.markdown("""
# #             ### How to Use:
# #             1. Upload an image containing traffic signs
# #             2. Adjust the confidence threshold if needed
# #             3. Click 'Analyze Image' to start detection
# #             4. View results and download the report
            
# #             ### Features:
# #             - Real-time traffic sign detection
# #             - Advanced analytics visualization
# #             - Detailed PDF report generation
# #             - Confidence score analysis
# #             - Multiple sign detection support
            
# #             ### Tips:
# #             - Use clear, well-lit images
# #             - Adjust confidence threshold for better results
# #             - Enable analytics for detailed insights
# #             """)
        
# #         # Add about section
# #         with st.expander("👩‍💻 About the Developer"):
# #             st.markdown("""
# #             ### Group 11
# #             Final Year Project
# #             SE Department
            
# #             ### Project Supervisor
# #             Dr. Ahmad Niaz
            
# #             ### Technology Stack:
# #             - YOLOv8 for Object Detection
# #             - Streamlit for Web Interface
# #             - OpenCV for Image Processing
# #             - Python for Backend Logic
# #             """)
        
# #         # Add version info
# #         st.sidebar.markdown("---")
# #         st.sidebar.markdown("Version 1.0.0")

# # def verify_model_files():
# #     """Verify if required model files exist"""
# #     if not os.path.exists("best.pt"):
# #         st.error("Error: Model file 'best.pt' not found! Please ensure the model file is in the same directory.")
# #         st.stop()

# # if __name__ == "__main__":
# #     try:
# #         verify_model_files()  # Check for model files before starting
# #         main()
# #     except Exception as e:
# #         st.error(f"An error occurred: {str(e)}")
# #         st.error("Please refresh the page and try again. If the problem persists, contact support.")