import os

import cv2
import matplotlib.pyplot as plt
import numpy
from emnist import extract_training_samples
from sklearn.neural_network import MLPClassifier

print("Imported the EMNIST libraries we need!")

# X will be our images and y will be the labels
X, y = extract_training_samples('letters')

# Make sure that every pixel in all of the images is a value between 0 and 1
X = X / 255.

# Use the first 60000 instances as training and the next 10000 as testing
X_train, X_test = X[:60000], X[60000:70000]
y_train, y_test = y[:60000], y[60000:70000]

# record the number of samples in each dataset and the number of pixels in each image
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

print("Extracted our samples and divided our training and testing data sets")

mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100,), max_iter=50, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)
mlp.fit(X_train, y_train)
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

# Puts all the data in the "files" variable
path, dirs, files = next(os.walk(r"C:\Users\Mothafucka\Desktop\Final_ML\LettersData\letters_mod/"))
files.sort()

# This code processes all the scanned images and adds them to the handwritten_story
handwritten_story = []
for i in range(len(files)):
    img = cv2.imread(r"C:\Users\Mothafucka\Desktop\Final_ML\LettersData\letters_mod/" + files[i], cv2.IMREAD_GRAYSCALE)
    handwritten_story.append(img)

print("Imported the scanned images.")

# These steps process the scanned images to be in the same format and have the same properties as the EMNIST images
processed_story = []

for img in handwritten_story:
    # step 1: Apply Gaussian blur filter
    img = cv2.GaussianBlur(img, (7, 7), 0)

    # steps 2 and 3: Extract the Region of Interest in the image and center in square
    points = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(points)
    if w > 0 and h > 0:
        if w > h:
            y = y - (w - h) // 2
            img = img[y:y + w, x:x + w]
        else:
            x = x - (h - w) // 2
            img = img[y:y + h, x:x + h]

    # step 4: Resize and resample to be 28 x 28 pixels
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)

    # step 5: Normalize pixels and reshape before adding to the new story array
    img = img / 255
    img = img.reshape((28, 28))
    processed_story.append(img)

print("Processed the scanned images.")

plt.imshow(processed_story[13])  # <<< change this index if you want to see a different letter from the story
plt.show()

typed_story = ""
for letter in processed_story:
    # this bit of code checks to see if the image is just a blank space by looking at the color of all the pixels summed
    total_pixel_value = 0
    for j in range(28):
        for k in range(28):
            total_pixel_value += letter[j, k]
    if total_pixel_value < 20:
        typed_story = typed_story + " "
    else:  # if it NOT a blank, it actually runs the prediction algorithm on it
        single_item_array = (numpy.array(letter)).reshape(1, 784)
        prediction = mlp.predict(single_item_array)
        typed_story = typed_story + str(chr(prediction[0] + 96))

print("Conversion to typed story complete!")
print(typed_story)
