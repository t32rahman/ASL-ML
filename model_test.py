import pickle

import cv2
import mediapipe as mp
import numpy as np

# Load the model from the pickle file
model_file = open('./model.p', 'rb')
model_dict = pickle.load(model_file)
model = model_dict['model']

# Open the webcam for capturing images
webcam = cv2.VideoCapture(0)

# Initialize the mediapipe hands module
hands_module = mp.solutions.hands
drawing_module = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles

hands = hands_module.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Create a dictionary for the labels of the classes (letters)
labels_dict = {}

for class_index in range(26):
    labels_dict[class_index] = chr(class_index + 65)

# Loop until the user quits
while True:

    # Create an empty list for the data of each image
    hand_data_item = []
    # Create empty lists for the x and y coordinates of the hand landmarks
    x_coords = []
    y_coords = []

    # Read a frame from the webcam
    ret, frame = webcam.read()

    # Get the height, width, and channels of the frame
    H, W, _ = frame.shape

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with the hands module
    results = hands.process(frame_rgb)
    # If hand landmarks are detected
    if results.multi_hand_landmarks:
        # Loop over the hand landmarks of each hand
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks and connections on the frame
            drawing_module.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                hands_module.HAND_CONNECTIONS,  # hand connections
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style())

        # Loop over the hand landmarks of each hand again
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

        # Get the coordinates of the bounding box of the hand
        x1 = int(min(x_coords) * W) - 10
        y1 = int(min(y_coords) * H) - 10

        x2 = int(max(x_coords) * W) - 10
        y2 = int(max(y_coords) * H) - 10

        # Predict the class (letter) of the hand
        prediction = model.predict([np.asarray(hand_data_item)])

        # Get the predicted character from the dictionary
        predicted_character = labels_dict[int(ord(prediction[0]) - 65)]

        # Draw a rectangle and a text on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    # Show the frame
    cv2.imshow('frame', frame)
    # Wait for a short time
    cv2.waitKey(1)


# Release the webcam and close the windows
webcam.release()
cv2.destroyAllWindows()
