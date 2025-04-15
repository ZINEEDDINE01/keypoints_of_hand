import os
import cv2
import mediapipe as mp
import csv

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define input and output directories
dataset_folder = "dataset_test"  # Change this to your dataset folder
output_folder = "dataset_test_keypoints"  # Folder to save annotated images
csv_file = "hand_keypoints.csv"  # CSV file to store keypoints

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Open CSV file for writing
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "hand", "keypoint_index", "x", "y", "z"])  # CSV header

    # Ensure dataset folder exists
    if not os.path.exists(dataset_folder):
        print(f"Error: Dataset folder '{dataset_folder}' does not exist!")
        exit()

    # Process all images in the folder
    for filename in os.listdir(dataset_folder):
        if filename.lower().endswith((".png", ".bmp", ".jpg")):
            image_path = os.path.join(dataset_folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Error: Could not read {filename}, skipping...")
                continue

            print(f"Processing {filename}...")

            # Convert image to RGB for MediaPipe processing
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            # Check if hands are detected
            if results.multi_hand_landmarks:
                print(f"Hands detected in {filename}")

                for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Draw landmarks on the image
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Write keypoints to CSV
                    for i, landmark in enumerate(hand_landmarks.landmark):
                        writer.writerow([filename, hand_index, i, landmark.x, landmark.y, landmark.z])

                # Save the processed image with keypoints
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, image)
                print(f"Saved annotated image: {output_path}")
            else:
                print(f"No hands detected in {filename}, skipping...")

print(f"\nProcessing completed. Hand keypoints saved to {csv_file}")
