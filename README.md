# 🧠 Brain Tumor Detection & Segmentation AI

This project is an advanced Machine Learning web application that detects the presence of brain tumors in MRI scans using Deep Learning and Computer Vision. 

## ✨ Features
* **Deep Learning Classification:** Utilizes Transfer Learning (MobileNetV2) for high-accuracy binary classification (Tumor vs. No Tumor).
* **Image Segmentation:** Implements the Watershed algorithm via OpenCV to visually isolate and highlight the tumor region.
* **Interactive Web UI:** Built with Streamlit for a seamless, user-friendly experience.

## 🛠️ Technologies Used
* **Python 3**
* **TensorFlow / Keras** (Model training and prediction)
* **OpenCV & Pillow** (Image processing and contour detection)
* **Streamlit** (Web framework)

## 🚀 How to Run Locally

1. Clone this repository:
    git clone https://github.com/sharathbandis/BrainTumor.git



3. Navigate to the project directory:
   cd YOUR_REPO_NAME




3. Install the required dependencies:
   pip install -r requirements.txt




5. Run the Streamlit application:
   python -m streamlit run app.py





## 📂 Project Structure

* `app.py`: The main Streamlit web application.
* `predict.py`: Handles image preprocessing (contour cropping) and model prediction.
* `displayTumor.py`: Contains the OpenCV Watershed segmentation logic.
* `train_model.py`: The script used to fine-tune the MobileNetV2 architecture.
* `models/`: Contains the trained `.h5` model weights.

