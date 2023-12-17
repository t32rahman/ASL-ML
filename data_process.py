import os
import pickle

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# Initialize the mediapipe hands module
hands_module = mp.solutions.hands
drawing_module = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles

hands = hands_module.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define the data directory
DATA_DIR = './data'

# Create empty lists for data and labels
hand_data = []
hand_labels = []
# Loop over the subdirectories in the data directory
for sub_dir in os.listdir(DATA_DIR):
    # Loop over the image files in each subdirectory
    for img_file in os.listdir(os.path.join(DATA_DIR, sub_dir)):
        # Create an empty list for the data of each image
        hand_data_item = []

        # Create empty lists for the x and y coordinates of the hand landmarks
        x_coords = []
        y_coords = []

        # Read the image file and convert it to RGB
        img = cv2.imread(os.path.join(DATA_DIR, sub_dir, img_file))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image with the hands module
        results = hands.process(img_rgb)
        # If hand landmarks are detected
        if results.multi_hand_landmarks:
            # Loop over the hand landmarks of each hand
            for hand_landmarks in results.multi_hand_landmarks:
                # Loop over the index and the landmark of each hand
                for index, landmark in enumerate(hand_landmarks.landmark):
                    # Get the x and y coordinates of the landmark
                    x = landmark.x
                    y = landmark.y

                    # Append the coordinates to the corresponding lists
                    x_coords.append(x)
                    y_coords.append(y)

                # Loop over the index and the landmark of each hand again
                for index, landmark in enumerate(hand_landmarks.landmark):
                    # Get the x and y coordinates of the landmark
                    x = landmark.x
                    y = landmark.y
                    # Append the normalized coordinates to the data item list
                    hand_data_item.append(x - min(x_coords))
                    hand_data_item.append(y - min(y_coords))

            # Append the data item and the label to the data and labels lists
            hand_data.append(hand_data_item)
            hand_labels.append(sub_dir)

# Open a pickle file for writing
pickle_file = open('data.pickle', 'wb')
# Dump the data and labels as a dictionary to the pickle file
pickle.dump({'data': hand_data, 'labels': hand_labels}, pickle_file)
# Close the pickle file
pickle_file.close()
