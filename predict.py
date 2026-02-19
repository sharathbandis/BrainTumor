import cv2 as cv
import numpy as np
import imutils
from tensorflow.keras.models import load_model

# Load the model once
MODEL_PATH = 'models/brain_tumor_detector.h5'
model = load_model(MODEL_PATH)

def crop_brain_contour(image):
    # Convert to grayscale and blur
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)

    # Threshold and morph
    thresh = cv.threshold(gray, 45, 255, cv.THRESH_BINARY)[1]
    thresh = cv.erode(thresh, None, iterations=2)
    thresh = cv.dilate(thresh, None, iterations=2)

    # Find contours
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    if not cnts:
        return image # Return original if no contours found

    c = max(cnts, key=cv.contourArea)

    # Find extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # Crop the image
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    return new_image

def predict_tumor(image):
    # 1. Crop the brain area
    cropped_img = crop_brain_contour(image)

    # 2. Resize and normalize for the model (matching your original logic)
    resized_img = cv.resize(cropped_img, dsize=(240, 240), interpolation=cv.INTER_CUBIC)
    normalized_img = resized_img / 255.0
    
    # 3. Reshape for prediction
    input_img = normalized_img.reshape((1, 240, 240, 3))

    # 4. Predict
    res = model.predict(input_img)
    return res[0][0] # Assuming it returns a 2D array like [[0.98]]