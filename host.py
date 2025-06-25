import streamlit as st
import cv2
import numpy as np
import time
import random
import os
from pygame import mixer
import tensorflow as tf 

model = tf.keras.models.load_model('cnn_model002.h5')

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

MUSIC_PATH = "MLProjSongs"

num_classes = model.output_shape[-1]
print(f"Model output classes: {num_classes}")

if num_classes != len(EMOTIONS):
    st.warning(f"Model expects {num_classes} classes, but EMOTIONS list has {len(EMOTIONS)}. Please update EMOTIONS.")

mixer.init()

def load_random_music(emotion):
    emotion_folder = os.path.join(MUSIC_PATH, emotion)
    music_files = [f for f in os.listdir(emotion_folder) if f.endswith('.mp3')]
    if music_files:
        return os.path.join(emotion_folder, random.choice(music_files))
    return None

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    input_data = np.expand_dims(normalized, axis=(0, -1))
    prediction = model.predict(input_data)
    emotion_idx = np.argmax(prediction)
    confidence = prediction[0][emotion_idx]
    if emotion_idx >= len(EMOTIONS):
        return "Unknown", confidence
    return EMOTIONS[emotion_idx], confidence

def main():
    st.set_page_config(page_title="Emotion Music Player", layout="wide")

    st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
    }
    .title {
        color: #2c3e50;
        font-size: 2.5em;
        text-align: center;
        margin-bottom: 20px;
    }
    .video-container {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stats-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    .emotion-text {
        font-size: 1.5em;
        color: #2980b9;
    }
    .timer-text {
        font-size: 1.2em;
        color: #e74c3c;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="title">Emotion-Based Music Player</h1>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        video_placeholder = st.empty()
    with col2:
        stats_placeholder = st.empty()
        timer_placeholder = st.empty()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    last_detection_time = time.time() - 5 
    current_emotion = None
    current_music = None
    confidence = 0.0
    DETECTION_INTERVAL = 5  

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        st.error("Error: Could not load face cascade classifier.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame from webcam.")
            break

        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
        current_time = time.time()
        elapsed_time = current_time - last_detection_time
        remaining_time = max(0, DETECTION_INTERVAL - elapsed_time)

        # Perform emotion detection every DETECTION_INTERVAL seconds
        if elapsed_time >= DETECTION_INTERVAL and len(faces) > 0:
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                emotion, conf = process_frame(face_roi)
                confidence = conf
                if emotion != current_emotion:
                    if mixer.music.get_busy():
                        mixer.music.stop()
                    music_file = load_random_music(emotion)
                    if music_file:
                        mixer.music.load(music_file)
                        mixer.music.play()
                        current_music = os.path.basename(music_file)
                        current_emotion = emotion
                    break
            last_detection_time = current_time

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if current_emotion:
                text = f"{current_emotion}: {confidence:.2f}"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        stats_html = f"""
        <div class="stats-box">
            <h3>Current Status</h3>
            <p class="emotion-text">Emotion: {current_emotion or 'Detecting...'}</p>
            <p>Confidence: {confidence:.2f}</p>
            <p>Playing: {current_music or 'None'}</p>
        </div>
        """
        stats_placeholder.markdown(stats_html, unsafe_allow_html=True)

        timer_html = f"""
        <p class="timer-text">Next Detection In: {remaining_time:.1f} seconds</p>
        """
        timer_placeholder.markdown(timer_html, unsafe_allow_html=True)

    cap.release()

if __name__ == "__main__":
    main()
