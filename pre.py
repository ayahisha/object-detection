from flask import Flask, send_file, jsonify
import cv2
import os
import numpy as np
import pyttsx3
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 200)  # Adjust speech speed

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Load dataset and train KNN
labels = []
features = []
class_names = []

dataset_path = "./dataset/"  # Adjust to your dataset path
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

knn = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree', weights='distance')
knn.fit(features, labels)

@app.route('/')
def index():
    return send_file("index.html")  # Serve index.html from the main directory

@app.route('/start_detection')
def start_detection():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (50, 50)).flatten().reshape(1, -1)
        prediction = knn.predict(resized)[0]
        detected_object = class_names[prediction]

        speak(f"Detected: {detected_object}")

        cv2.putText(frame, detected_object, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return jsonify({"message": "Detection Stopped"})

if __name__ == '__main__':
    app.run(debug=True)
