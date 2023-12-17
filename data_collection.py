import os
import cv2

# Define the data directory and create it if it does not exist
DATA_DIR = './data'
os.makedirs(DATA_DIR, exist_ok=True)

# Define the number of classes (letters) and the size of the dataset (images) per class
NUM_CLASSES = 26
DATASET_SIZE = 120

# Open the webcam for capturing images
webcam = cv2.VideoCapture(1)

# Loop over the classes (letters)
for class_index in range(NUM_CLASSES):
    # Get the letter corresponding to the class index
    letter = chr(class_index + 65)
    # Define the subdirectory for the letter and create it if it does not exist
    letter_subdir = os.path.join(DATA_DIR, letter)
    os.makedirs(letter_subdir, exist_ok=True)

    # Print the message for the class (letter)
    print(f'Collecting data for class {letter}')

    # Wait for the user to press "Q" to start capturing images
    while True:
        # Read a frame from the webcam
        ret, frame = webcam.read()
        # Put a text on the frame
        cv2.putText(frame, f'Press "Q" to start capturing {letter}', (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3,
                    cv2.LINE_AA)
        # Show the frame
        cv2.imshow('frame', frame)
        # Break the loop if the user presses "Q"
        if cv2.waitKey(25) == ord('q'):
            break

    # Initialize the image counter
    image_counter = 0
    # Loop until the dataset size is reached
    while image_counter < DATASET_SIZE:
        # Read a frame from the webcam
        ret, frame = webcam.read()
        # Show the frame
        cv2.imshow('frame', frame)
        # Wait for a short time
        cv2.waitKey(33)
        # Save the frame as an image file
        cv2.imwrite(os.path.join(letter_subdir, f'{image_counter}.jpg'), frame)

        # Increment the image counter
        image_counter += 1

# Release the webcam and close the windows
webcam.release()
cv2.destroyAllWindows()