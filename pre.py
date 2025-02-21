from flask import Flask, Response, jsonify, send_file
import cv2
import os
import numpy as np
import pyttsx3
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 200)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Load dataset and train KNN
labels = []
features = []
class_names = []

dataset_path = "./dataset/"  # Adjust dataset path
if os.path.exists(dataset_path):
    folders = os.listdir(dataset_path)

    for i, folder in enumerate(folders):
        class_names.append(folder)
        folder_path = os.path.join(dataset_path, folder)
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (50, 50)).flatten()
            features.append(img)
            labels.append(i)

features = np.array(features)
labels = np.array(labels)

if len(features) > 0:
    knn = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree', weights='distance')
    knn.fit(features, labels)
else:
    knn = None  # Avoid errors if dataset is missing

camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (50, 50)).flatten().reshape(1, -1)

        if knn:
            prediction = knn.predict(resized)[0]
            detected_object = class_names[prediction]
            speak(f"Detected: {detected_object}")

            # Draw bounding box
            h, w, _ = frame.shape
            cv2.rectangle(frame, (50, 50), (w-50, h-50), (0, 255, 0), 3)
            cv2.putText(frame, detected_object, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return send_file("index.html")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_detection')
def stop_detection():
    global camera
    camera.release()
    return jsonify({"message": "Detection Stopped"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
