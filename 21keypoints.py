import os
import cv2
import mediapipe as mp
import csv

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define relative paths
dataset_folder = "dataset_test"
csv_file = "cam1.csv"
csv_folder = os.path.dirname(csv_file)  # Extract directory path

# Ensure the directory exists **only if csv_file is inside a folder**
if csv_folder and not os.path.exists(csv_folder):
    print(f"Directory '{csv_folder}' does not exist. Creating it now...")
    os.makedirs(csv_folder)

# Open CSV file for writing
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "hand", "keypoint_index", "x", "y", "z"])  # CSV header

    # Ensure dataset folder exists
    if not os.path.exists(dataset_folder):
        print(f"Error: Dataset folder '{dataset_folder}' does not exist!")
        exit()

    for filename in os.listdir(dataset_folder):
        if filename.lower().endswith((".png", ".bmp")):
            image_path = os.path.join(dataset_folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Error: Could not read {filename}, skipping...")
                continue

            print(f"Processing {filename}...")

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                print(f"Hands detected in {filename}")
                for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    for i, landmark in enumerate(hand_landmarks.landmark):
                        writer.writerow([filename, hand_index, i, landmark.x, landmark.y, landmark.z])
            else:
                print(f"No hands detected in {filename}, skipping...")

print(f"Hand keypoints saved to {csv_file}")
