import customtkinter as ctk
import cv2
import os
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tkinter import filedialog, messagebox
from PIL import Image

# Set appearance mode and theme
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("dark-blue")

# Constants
DATASET_PATH = "dataset"
IMAGE_SIZE = (32, 32)
MODEL_FILE = "knn_model.pkl"

def train_model():
    """Function to train the model, calculate accuracy, and save it."""
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

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Split dataset for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=50)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Train KNN model
    knn = KNeighborsClassifier(n_neighbors=3, algorithm="ball_tree")
    knn.fit(X_train_pca, y_train)

    # Calculate accuracy
    y_pred = knn.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)

    # Save the model
    with open(MODEL_FILE, "wb") as model_file:
        pickle.dump((knn, pca, labels), model_file)

    messagebox.showinfo("Training Complete", f"Model training completed successfully!\nAccuracy: {accuracy * 100:.2f}%")

def recognize_sign():
    """Function to recognize signs via webcam."""
    if not os.path.exists(MODEL_FILE):
        messagebox.showerror("Error", "Model not found. Train the model first.")
        return

    with open(MODEL_FILE, "rb") as model_file:
        knn, pca, labels = pickle.load(model_file)

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, IMAGE_SIZE).flatten()
        gray_pca = pca.transform([gray])
        label_id = knn.predict(gray_pca)[0]
        label = labels[label_id]

        cv2.putText(frame, f"Prediction: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Sign Language Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def upload_and_recognize():
    """Function to upload an image and recognize the sign."""
    if not os.path.exists(MODEL_FILE):
        messagebox.showerror("Error", "Model not found. Train the model first.")
        return

    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return  # User canceled

    with open(MODEL_FILE, "rb") as model_file:
        knn, pca, labels = pickle.load(model_file)

    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMAGE_SIZE).flatten()
    img_pca = pca.transform([img])
    label_id = knn.predict(img_pca)[0]
    label = labels[label_id]

    # Display result
    img = Image.open(file_path)
    img.show()
    messagebox.showinfo("Recognition Result", f"Predicted Sign: {label}")

# Initialize GUI
app = ctk.CTk()
app.title("Sign Language Recognition")
app.geometry("800x500")

# Sidebar
sidebar_frame = ctk.CTkFrame(app, width=200, corner_radius=15)
sidebar_frame.pack(side="left", fill="y", padx=10, pady=10)

sidebar_label = ctk.CTkLabel(sidebar_frame, text="Sign Language", font=("Arial", 16, "bold"))
sidebar_label.pack(pady=(10, 20))

button_train = ctk.CTkButton(sidebar_frame, text="Train Model", command=train_model)
button_train.pack(pady=10)

button_recognition = ctk.CTkButton(sidebar_frame, text="Start Webcam Recognition", command=recognize_sign)
button_recognition.pack(pady=10)

button_upload = ctk.CTkButton(sidebar_frame, text="Upload & Recognize", command=upload_and_recognize)
button_upload.pack(pady=10)

# Main Content Frame
content_frame = ctk.CTkFrame(app, corner_radius=15)
content_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

content_label = ctk.CTkLabel(content_frame, text="Sign Language Recognition", font=("Arial", 20, "bold"))
content_label.pack(pady=(20, 10))

camera_placeholder = ctk.CTkFrame(content_frame, height=300, corner_radius=15)
camera_placeholder.pack(fill="x", padx=20, pady=(10, 20))

camera_label = ctk.CTkLabel(camera_placeholder, text="Camera View Will Appear Here", font=("Arial", 16))
camera_label.pack(expand=True)

# Run the Application
app.mainloop()
