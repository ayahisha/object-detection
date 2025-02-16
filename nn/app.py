from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import pickle
from sklearn.decomposition import PCA

app = Flask(__name__)
CORS(app)

# Load trained KNN model
with open("knn_model.pkl", "rb") as model_file:
    knn, pca, labels = pickle.load(model_file)

IMAGE_SIZE = (64, 64)
cap = cv2.VideoCapture(0)  # Open the webcam

def detect_objects():
    """Continuously capture frames, process them, and detect objects."""
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, IMAGE_SIZE).flatten().reshape(1, -1)
        transformed = pca.transform(resized)
        label_id = knn.predict(transformed)[0]
        label_name = labels[label_id]

        # Draw a bounding box around the detected object
        height, width = frame.shape[:2]
        start_x, start_y, end_x, end_y = width//3, height//3, (2*width)//3, (2*height)//3
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        cv2.putText(frame, label_name, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route("/video_feed")
def video_feed():
    """Video streaming route."""
    return Response(detect_objects(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/detect", methods=["GET"])
def detect():
    """Return detected object label."""
    ret, frame = cap.read()
    if not ret:
        return jsonify({"error": "Camera not available"}), 500

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMAGE_SIZE).flatten().reshape(1, -1)
    transformed = pca.transform(resized)
    label_id = knn.predict(transformed)[0]
    label_name = labels[label_id]

    return jsonify({"detected_object": label_name})

if __name__ == "__main__":
    app.run(debug=True)
