import pickle

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the data and labels from the pickle file
pickle_file = open('./data.pickle', 'rb')
data_dict = pickle.load(pickle_file)
hand_data = np.asarray(data_dict['data'])
hand_labels = np.asarray(data_dict['labels'])

# Split the data and labels into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(hand_data, hand_labels, test_size=0.2, shuffle=True, stratify=hand_labels)

# Create a random forest classifier
classifier = RandomForestClassifier()

# Train the classifier on the training set
classifier.fit(train_data, train_labels)

# Predict the labels on the testing set
test_predict = classifier.predict(test_data)

# Calculate the accuracy score of the prediction
accuracy = accuracy_score(test_predict, test_labels)

# Print the accuracy percentage
print(f'{accuracy * 100}% of samples were classified correctly !')

# Save the classifier to a pickle file
model_file = open('model.p', 'wb')
pickle.dump({'model': classifier}, model_file)
model_file.close()
