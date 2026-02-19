import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
from predict import predict_tumor
from displayTumor import DisplayTumor # Importing your existing segmentation logic!

st.set_page_config(page_title="Brain Tumor Detector", page_icon="🧠", layout="wide")

st.title("🧠 Advanced Brain Tumor Detection & Segmentation")
st.write("Upload an MRI scan to analyze it using deep learning and watershed segmentation.")

uploaded_file = st.file_uploader("Choose an MRI image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Set up two columns for a beautiful UI
    col1, col2 = st.columns(2)
    
    image = Image.open(uploaded_file)
    with col1:
        st.subheader("Original MRI Scan")
        st.image(image, use_container_width=True)
    
    # Convert for OpenCV
    image_np = np.array(image)
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_cv = cv.cvtColor(image_np, cv.COLOR_RGB2BGR)
    else:
        image_cv = cv.cvtColor(image_np, cv.COLOR_GRAY2BGR)

    if st.button("Analyze MRI Scan", type="primary"):
        with st.spinner('Running AI Model & Segmentation...'):
            
            # 1. Get Prediction
            prediction_score = predict_tumor(image_cv)
            
            # 2. Get Visual Segmentation (Using your code!)
            dt = DisplayTumor()
            dt.readImage(image_cv)
            dt.removeNoise()
            dt.displayTumor()
            segmented_img_bgr = dt.getImage()
            # Convert BGR back to RGB for Streamlit display
            segmented_img_rgb = cv.cvtColor(segmented_img_bgr, cv.COLOR_BGR2RGB)
            
            with col2:
                st.subheader("Tumor Segmentation Map")
                st.image(segmented_img_rgb, use_container_width=True)

            st.markdown("---")
            if prediction_score > 0.5:
                st.error(f"⚠️ **Tumor Detected** (AI Confidence: {prediction_score*100:.2f}%)")
                st.write("The red highlighted region in the segmentation map indicates the suspected tumor area.")
            else:
                st.success(f"✅ **No Tumor Detected** (AI Confidence: {(1-prediction_score)*100:.2f}%)")