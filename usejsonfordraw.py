import os
import cv2
import mediapipe as mp
import json

def generate_json(dataset_folder="dataset_test", json_output="hand_keypoints.json"):
    """
    Process images with MediaPipe to extract hand keypoints and save the data as JSON.
    
    The JSON structure is:
    {
      "images": [
         {
           "filename": "image1.jpg",
           "hands": [
              {
                "hand_index": 0,
                "landmarks": [
                  {"x": ..., "y": ..., "z": ...},  <-- 21 items total
                  ...
                ]
              },
              ... (if multiple hands detected)
           ]
         },
         ... (other images)
      ]
    }
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    )
    
    data = {"images": []}
    
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

            # Convert the image to RGB for MediaPipe processing.
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            image_hands = []

            if results.multi_hand_landmarks:
                for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    hand_data = {"hand_index": hand_index, "landmarks": []}
                    # Save all 21 keypoints
                    for landmark in hand_landmarks.landmark:
                        hand_data["landmarks"].append({
                            "x": landmark.x,  # normalized coordinate (0 to 1)
                            "y": landmark.y,
                            "z": landmark.z
                        })
                    image_hands.append(hand_data)

            data["images"].append({
                "filename": filename,
                "hands": image_hands
            })
    
    # Write the JSON file.
    with open(json_output, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Hand keypoints JSON saved to '{json_output}'")
    return data


def draw_keypoints_from_json(data, dataset_folder="dataset_test", output_folder="dataset_test_keypoints"):
    """
    Draws the 21 keypoints and skeleton connections on each image based on the JSON data.
    
    The drawn keypoints are marked with red circles and the skeletal connections are drawn with white lines.
    """
    # MediaPipe canonical hand connections for the 21 keypoints.
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20)
    ]
    
    os.makedirs(output_folder, exist_ok=True)
    
    for image_data in data.get("images", []):
        filename = image_data.get("filename")
        image_path = os.path.join(dataset_folder, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {filename}")
            continue

        height, width = image.shape[:2]

        for hand in image_data.get("hands", []):
            landmarks = hand.get("landmarks", [])
            if len(landmarks) != 21:
                # Skip if the hand doesn't have exactly 21 keypoints.
                continue

            # Convert normalized coordinates to pixel positions.
            landmark_pixels = []
            for kp in landmarks:
                x_px = int(kp["x"] * width)
                y_px = int(kp["y"] * height)
                landmark_pixels.append((x_px, y_px))
                # Draw a red circle for each keypoint.
                cv2.circle(image, (x_px, y_px), 5, (0, 0, 255), thickness=-1)

            # Draw skeletal connection lines.
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(landmark_pixels) and end_idx < len(landmark_pixels):
                    cv2.line(image, landmark_pixels[start_idx], landmark_pixels[end_idx], (255, 255, 255), thickness=2)

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, image)
        print(f"Annotated image saved to '{output_path}'")
    
    print("All images have been processed and annotated with hand keypoints.")


def main():
    # Configuration for input and output directories, and JSON file.
    dataset_folder = "dataset_test"             # Folder containing input images.
    output_folder = "dataset_test_keypoints"      # Folder to save annotated images.
    json_file = "hand_keypoints.json"             # JSON filename.

    # Step 1: Generate the JSON file with hand keypoints.
    data = generate_json(dataset_folder, json_file)
    
    # Step 2: Draw keypoints on images using the generated JSON data.
    draw_keypoints_from_json(data, dataset_folder, output_folder)


if __name__ == "__main__":
    main()
