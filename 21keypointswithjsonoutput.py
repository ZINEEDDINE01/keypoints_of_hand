import os
import cv2
import mediapipe as mp
import json

# Initialize MediaPipe Hands solution.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5)
# (Optional) Use mp.solutions.drawing_utils if you want to draw on images here.

# Define the folder containing your images.
dataset_folder = "dataset_test"   # Folder that contains your input images.
json_output = "hand_keypoints.json"  # JSON output filename.

# Data container that will be saved as JSON.
# The JSON structure:
# {
#   "images": [
#       {
#         "filename": "image1.jpg",
#         "hands": [
#           {
#             "hand_index": 0,
#             "landmarks": [
#               {"x": ..., "y": ..., "z": ...},  <-- 21 items total
#               ... (total 21 keypoints)
#             ]
#           },
#           ... (if multiple hands detected)
#         ]
#       },
#       ... (other images)
#   ]
# }
data = {"images": []}

# Process each image in the dataset folder.
if not os.path.exists(dataset_folder):
    print(f"Error: The dataset folder '{dataset_folder}' does not exist!")
    exit()

for filename in os.listdir(dataset_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        image_path = os.path.join(dataset_folder, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {filename}")
            continue

        # Convert image from BGR to RGB for MediaPipe processing.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        image_hands = []

        if results.multi_hand_landmarks:
            for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_data = {"hand_index": hand_index, "landmarks": []}
                # For each of the 21 keypoints.
                for landmark in hand_landmarks.landmark:
                    hand_data["landmarks"].append({
                        "x": landmark.x,  # normalized x (0 to 1)
                        "y": landmark.y,  # normalized y (0 to 1)
                        "z": landmark.z   # normalized z (depth; not used for drawing)
                    })
                image_hands.append(hand_data)

        # Save the image data (even if no hand is detected, hands list may be empty).
        data["images"].append({
            "filename": filename,
            "hands": image_hands
        })

# Write the collected data to a JSON file.
with open(json_output, "w") as f:
    json.dump(data, f, indent=2)

print(f"Hand keypoints JSON saved to '{json_output}'")