import os
import cv2
import numpy as np
import pickle
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image

app = Flask(__name__)
app.secret_key = "secret_key"

# Constants
UPLOAD_FOLDER = "static/uploads"
DATASET_PATH = "dataset"
IMAGE_SIZE = (32, 32)
MODEL_FILE = "knn_model.pkl"

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/train", methods=["POST"])
def train_model():
    """Function to train the model and save it."""
    X, y = [], []
    labels = {}
    label_id = 0

    # Load dataset
    for folder in os.listdir(DATASET_PATH):
        folder_path = os.path.join(DATASET_PATH, folder)
        if os.path.isdir(folder_path):
            labels[label_id] = folder
            for file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, IMAGE_SIZE)
                X.append(img.flatten())
                y.append(label_id)
            label_id += 1

    X = np.array(X)
    y = np.array(y)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply PCA
    pca = PCA(n_components=50)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Train KNN Model
    knn = KNeighborsClassifier(n_neighbors=3, algorithm="ball_tree")
    knn.fit(X_train_pca, y_train)

    # Evaluate Model
    y_pred = knn.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)

    # Save Model
    with open(MODEL_FILE, "wb") as model_file:
        pickle.dump((knn, pca, labels), model_file)

    flash(f"Model trained successfully! Accuracy: {accuracy * 100:.2f}%", "success")
    return redirect(url_for("home"))

@app.route("/upload", methods=["POST"])
def upload_and_recognize():
    """Function to recognize a sign from an uploaded image."""
    if "file" not in request.files:
        flash("No file uploaded!", "error")
        return redirect(url_for("home"))

    file = request.files["file"]
    if file.filename == "":
        flash("No selected file!", "error")
        return redirect(url_for("home"))

    if not os.path.exists(MODEL_FILE):
        flash("Model not found! Train the model first.", "error")
        return redirect(url_for("home"))

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Load Model
    with open(MODEL_FILE, "rb") as model_file:
        knn, pca, labels = pickle.load(model_file)

    # Process Image
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMAGE_SIZE).flatten()
    img_pca = pca.transform([img])
    label_id = knn.predict(img_pca)[0]
    label = labels[label_id]

    return render_template("result.html", label=label, image_path=filepath)

if __name__ == "__main__":
    app.run(debug=True)
