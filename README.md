# Final_ML
author: Jacob Fiske
Class: CS3120
Final Machine Learning Project: Ai Drawing Letter Prediction

Resources:
Packages - OpenCV, MediaPipe, SKlearn, Emnist, Numpy
Links - (Frequently used documentation)
        https://docs.opencv.org/master/
        https://google.github.io/mediapipe/
        (Supplemental Learning)
        Murtaza's Workshop - Robotics and AI
        (CS3120 Lecture Videos and Notes)

Steps:
!!! IMPORTANT A WEBCAM OR WIRED CAMERA IS REQUIRED TO RUN THE CODE !!!
Configure and run "recognition.py"
two windows will pop up: the black canvas window and the camera.
the user must show their hand with their palm facing the camera.
(use a well lit uniform background for best results, from my experience)
Once the hand is recognized the user can start to draw.
Hold the index finger and middle finger up with the rest of the fingers down to pause drawing.
Hold only the index finger up to start drawing.
Once finished press q to save the handwritten letter for the MLP to start training then predict
the given image.
The iterations, loss, score, and prediction are outputted into the console

