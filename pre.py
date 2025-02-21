from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# Load KNN model
labels = []
features = []
class_names = []

dataset_path = "./dataset/"
if dataset_path:
    for i, folder in enumerate(["tv", "table", "street", "persons", "fridge", "chair", "almirah"]):
        class_names.append(folder)

knn = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree', weights='distance')

@app.route('/')
def index():
    return open("index.html").read()

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json['image']
    img_data = base64.b64decode(data.split(',')[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    
    resized = cv2.resize(frame, (50, 50)).flatten().reshape(1, -1)
    detected_object = "Unknown"
    
    if knn:
        prediction = knn.predict(resized)[0]
        detected_object = class_names[prediction]

    return jsonify({'object': detected_object})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
