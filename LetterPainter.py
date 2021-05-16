import cv2
import numpy as np
import HandTrackingModule as htm

# Creates a canvas to draw on using the index and middle finger.
# When both the middle and index finger are up drawing is paused.
# When only the index finger is up one can draw on the canvas and recording camera.


# initializes video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Initializes hand detector with a high detection cons
detector = htm.handDetector(detectionCon=.85)
xp, yp = 0, 0

# creates a seperate image canvas to copy the handwritten letter onto
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

# starts app
while True:
    success, img = cap.read()

    # initializes image and position list
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:

        # Keeps track of index and middle finger positions
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # Checks if fingers are up
        fingers = detector.fingersUp()

        # Specifically checks if index and middle finger are both up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.rectangle(img, (x1, y1 - 15), (x2, y2 + 15), (255, 255, 255), cv2.FILLED)

        # Specifically checks if the index finger is up and the middle finger is down
        if fingers[1] and fingers[2] == False:
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            # draws lines from the index finger
            cv2.line(img, (xp, yp), (x1, y1), (255, 255, 255), 15)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), (255, 255, 255), 15)

            # updates positions
            xp, yp = x1, y1

    # Shows windows and starts process
    key = cv2.waitKey(1) & 0xFF
    img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)

    # Press 'q' once finished drawing a letter to initiate "recognition.py"
    if key == ord('q'):
        cv2.imwrite("RESULT.png", imgCanvas)
        break
