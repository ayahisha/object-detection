import customtkinter as ctk
import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import pickle
from tkinter import messagebox

def train_model():
    DATASET_PATH = "dataset"
    IMAGE_SIZE = (32, 32)

    X, y = [], []
    labels = {}
    label_id = 0

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

    # Apply PCA to reduce dimensions
    pca = PCA(n_components=50)
    X = pca.fit_transform(X)

    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=3, algorithm="ball_tree")
    knn.fit(X, y)

    # Save Model
    with open("knn_model.pkl", "wb") as model_file:
        pickle.dump((knn, pca, labels), model_file)

messagebox.showinfo("Training Complete", "Model training completed successfully!")

def open_camera():
    # Placeholder for camera functionality
    ctk.CTkMessagebox(title="Camera", message="Camera function not implemented yet.")

# Initialize CustomTkinter
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("dark-blue")

app = ctk.CTk()
app.title("Sign Language Recognition")
app.geometry("800x500")

# Sidebar Frame
sidebar_frame = ctk.CTkFrame(app, width=200, corner_radius=15)
sidebar_frame.pack(side="left", fill="y", padx=10, pady=10)

sidebar_label = ctk.CTkLabel(sidebar_frame, text="Sign Language", font=("Arial", 16, "bold"))
sidebar_label.pack(pady=(10, 20))

button_train = ctk.CTkButton(sidebar_frame, text="Train Model", command=train_model)
button_train.pack(pady=10)

button_recognition = ctk.CTkButton(sidebar_frame, text="Start Recognition", command=open_camera)
button_recognition.pack(pady=10)

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
