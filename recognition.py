import numpy
from LetterPainter import *
from emnist import extract_training_samples
from sklearn.neural_network import MLPClassifier

print("Imported the EMNIST libraries we need!")

# X will be our images and y will be the labels
X, y = extract_training_samples('letters')

# Scale values between 0 and 1
X = X / 255.

# 60,000 as training, 10,000 as testing
X_train, X_test = X[:60000], X[60000:70000]
y_train, y_test = y[:60000], y[60000:70000]

# record the number of samples in each dataset and the number of pixels in each image
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

print("Extracted our samples and divided our training and testing data sets")

# Create Multi Layer Perception Model using SKLEARN's model
# Goes through 20 iterations unless no progress is made, loss is calculated for each iteration
mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100,), max_iter=20, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)
# Fits our data
mlp.fit(X_train, y_train)

# Prints training and Test scores
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

# Final letter drawing result from LetterPainter
file = "./RESULT.png"

# Process result for a better outcome (emnist doc)
file = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
img = cv2.GaussianBlur(file, (7, 7), 0)

# Extract the Region of Interest in the image and center in square
points = cv2.findNonZero(img)
x, y, w, h = cv2.boundingRect(points)
if w > 0 and h > 0:
    if w > h:
        y = y - (w - h) // 2
        img = img[y:y + w, x:x + w]
    else:
        x = x - (h - w) // 2
        img = img[y:y + h, x:x + h]

# resize to fit data (28x28)
img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)

# Scale the pixels and reshape
img = img / 255
img = img.reshape((28, 28))

# run prediction and print the predicted letter
img = (numpy.array(img)).reshape(1, 784)
prediction = mlp.predict(img)
print("Predicited Result: " + str(chr(prediction[0] + 96)))
