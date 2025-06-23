# AI Drawing Letter Prediction – Final Machine Learning Project

**Author**: Jacob Fiske  
**Course**: CS3120 – Machine Learning  

## Project Overview
This project leverages computer vision and machine learning to recognize handwritten letters drawn in the air using a webcam. The user can draw letters with their hand, which are then processed and classified by a trained machine learning model.

## Dependencies
The following Python packages are required to run this project:

- `opencv-python`
- `mediapipe`
- `scikit-learn`
- `numpy`
- `emnist` (dataset)

Install them using pip:

```bash
pip install opencv-python mediapipe scikit-learn numpy
```
Note: You may need to manually download and load the EMNIST dataset if not included.
Reference Materials
Documentation

    OpenCV Documentation

    MediaPipe Documentation

Learning Resources

    Murtaza's Workshop – Robotics and AI (YouTube)

    CS3120 Lecture Videos and Notes

Setup & Usage Instructions

    ⚠️ Important: A webcam or wired camera is required to run this project.

Running the Program

    Execute the script:

    python recognition.py

    Two windows will appear:

        Canvas Window – A black screen where your hand-drawn letter appears.

        Camera Feed – Live feed for hand tracking.

    Drawing Controls:

        Show your hand with your palm facing the camera.

        Use a well-lit, uniform background for best results.

        Raise only the index finger to start drawing.

        Raise both the index and middle fingers (with others down) to pause drawing.

    Once finished, press q to:

        Save the image

        Train the MLP model

        Predict the drawn character

    Output in the console:

        Number of training iterations

        Loss value

        Accuracy score

        Final character prediction


