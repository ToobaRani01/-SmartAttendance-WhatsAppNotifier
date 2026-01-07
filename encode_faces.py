# Install prerequisites for face recognition

# pip install cmake
# pip install dlib
# pip install face_recognition
# import libraries
import face_recognition
import pandas as pd
import numpy as np
import os
import cv2
import pickle
from PIL import Image , ImageEnhance
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
# Step 1: load dataset
df= pd.read_csv("dataset.csv")
df.head()
print(df.columns)

# Step 0: Initialize FaceNet model and MTCNN for detection
# MTCNN is used for robust face detection and alignment
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

# InceptionResnetV1 (FaceNet-like architecture) pretrained on VGGFace2
# This model outputs 512-dimensional face embeddings
resnet = InceptionResnetV1(pretrained='vggface2').eval() # Set to eval mode for inference



# Step 2 : encodings and labels
known_encodings = []
known_names = []

# Loop through each student entry
for index, row in df.iterrows():
    student_id = row['Student_ID']
    name = row['Name']
    image_dir = row['Image_Dir']  # e.g., images/001_toobarani/

    # Go through each image of the student
    for img_name in os.listdir(image_dir):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(image_dir, img_name)

            # Step 2.1: Load image
            try:
                # MTCNN expects PIL Image
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"❌ Error loading {img_path}: {e}")
                continue

            # Step 2.2: Detect faces and get aligned images using MTCNN
            # MTCNN returns a list of cropped and aligned face tensors
            face_images = mtcnn(image)

            if face_images is None:
                print(f"No face detected in {img_path}")
                continue
            for face_img_tensor in face_images:
                face_embedding = resnet(face_img_tensor.unsqueeze(0)).detach().numpy().flatten()

                if face_embedding is not None:
                    known_encodings.append(face_embedding)
                    known_names.append(f"{student_id}_{name}")
                else:
                    print(f"No embedding generated for a face in {img_path}")


print(f"✅ Total faces encoded: {len(known_encodings)}")
# 💾 Save encodings
with open("encodings.pkl", "wb") as f:
    pickle.dump((known_encodings, known_names), f)

print("✅ Face encodings saved to encodings.pkl")