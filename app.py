import streamlit as st
import cv2
import joblib
import numpy as np
import os
from src.hand_tracking import HandTracker

st.set_page_config(page_title="Sign Language App", page_icon="🤟", layout="centered")

st.title("🤟 Real-Time Sign Language Translator")
st.markdown("Click the checkbox below to turn on your webcam and start translating.")

@st.cache_resource
def load_model():
    model_path = 'models/sign_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

if model is None:
    st.error("🚨 Model not found! Ensure 'sign_model.pkl' is inside the 'models' folder.")
else:
    st.success("✅ Machine Learning Model loaded successfully!")
    
    run_webcam = st.checkbox("Start Webcam")
    frame_placeholder = st.empty()
    
    if run_webcam:
        cap = cv2.VideoCapture(0)
        tracker = HandTracker(max_hands=1)
        
        while run_webcam:
            ret, frame = cap.read()
            if not ret: break
                
            frame = cv2.flip(frame, 1)
            frame = tracker.find_hands(frame)
            landmarks = tracker.get_landmarks(frame)
            
            if len(landmarks) == 63:
                # Create feature names to match training data
                import warnings
                warnings.filterwarnings('ignore')
                feature_names = [f'landmark_{i}_dim_{j}' for i in range(21) for j in range(3)]
                input_data = np.array(landmarks).reshape(1, -1)
                prediction = model.predict(input_data)[0]
                
                cv2.rectangle(frame, (0, 0), (300, 70), (0, 0, 0), cv2.FILLED)
                cv2.putText(frame, f"Sign: {prediction}", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")
            
        cap.release()