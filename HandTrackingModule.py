import cv2
import mediapipe as mp

# Creates a module to track the hand and fingers for multiple uses

# 0 = Wrist
# 4 = Thumb Tip
# 8 = Index Tip
# 12 = Middle Tip
# 16 = Ring Tip
# 20 = Pinky Tip

# Initializes handDetector
class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=.5, trackCon=.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

        self.tipIds = [4, 8, 12, 16, 20]

    # Detects hands and draws connections for landmarks
    def findHands(self, img, draw=True):
        # Converts to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    # finds the position of parts of the hand in a list format
    def findPosition(self, img, handNo=0, draw=True):

        self.lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return self.lmList

    # checks to see which fingers are up and which are down
    def fingersUp(self):
        fingers = []

        # Checks and stores whether or not the thumb is up
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Checks if every finger is up except for the thumb then stores
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

# Initiates video capture and hand detector and collects finger positions
def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        currentHand = detector.findPosition(img)
        if len(currentHand) != 0:
            print(currentHand[4])  # edit for different Tips

        cv2.imshow("Image ", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
