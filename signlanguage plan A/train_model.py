import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import pickle

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

print("Model training complete and saved as knn_model.pkl!")
