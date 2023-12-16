import os
import cv2 as cv
from pathlib import Path

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26
dataset_size = 120

cap = cv.VideoCapture(0)
for j in range(number_of_classes):
    letter = chr(j + 65)
    if not os.path.exists(os.path.join(DATA_DIR, letter)):
        os.makedirs(os.path.join(DATA_DIR, letter))

    print('Collecting data for class {}'.format(letter))

    done = False
    while True:
        ret, frame = cap.read()
        cv.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv.LINE_AA)
        cv.imshow('frame', frame)
        if cv.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv.imshow('frame', frame)
        cv.waitKey(33)
        cv.imwrite(os.path.join(DATA_DIR, letter, '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv.destroyAllWindows()