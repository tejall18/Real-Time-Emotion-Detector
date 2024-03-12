import tkinter as tk
from tkinter import filedialog, Button, Label
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

# Load the facial expression model
def FacialExpressionModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Initialize Tkinter
top = tk.Tk()
top.geometry('800x600')
top.title('Real-Time Emotion Detector')
top.configure(background='#CDCDCD')

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the trained model
model = FacialExpressionModel("model_RTED1.json", "model_weights_RTED1.h5")

# List of emotions
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Function to detect emotion in each frame
def detect_emotion():
    # Capture video from webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))

            # Predict emotion
            roi = roi_gray[np.newaxis, :, :, np.newaxis] / 255.0
            pred = EMOTIONS_LIST[np.argmax(model.predict(roi))]

            cv2.putText(frame, pred, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Real-Time Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Button to start emotion detection
detect_button = Button(top, text="Start Emotion Detection", command=detect_emotion, padx=10, pady=5)
detect_button.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
detect_button.pack(side='bottom', pady=50)

top.mainloop()
