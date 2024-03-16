from cvzone.HandTrackingModule import HandDetector
import cv2
import os
import numpy as np

# Parameters
width, height = 30000, 15000  # Width and height of the video frame
gestureThreshold = 850  # Gesture priority threshold (height). Action will be taken when hand's height crosses this threshold.
folderPath = "Presentation"  # Path for containing presentation images

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Hand detector
detectorHand = HandDetector(detectionCon=0.8, maxHands=1)

# Variables
imgList = []
delay = 30  # Delay time (in frames) to assist in action execution
buttonPressed = False  # Has any button been pressed by the hand?
counter = 0
drawMode = False  # Is in drawing mode?
imgNumber = 0  # Index of the current image in the presentation images list
delayCounter = 0  # Delay counter
annotations = [[]]  # List of annotations
annotationNumber = -1  # Index of the current annotation in the annotations list
annotationStart = False  # Has annotation started?
hs, ws = int(120 * 1), int(213 * 1)  # Width and height of the small image

# Get the list of presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
print(pathImages)

while True:
    # Get the image frame
    success, img = cap.read()
    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)

    # Find hands and their landmarks
    hands, img = detectorHand.findHands(img)
    # Draw the gesture priority line
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10) # Green

    if hands and buttonPressed is False:
        hand = hands[0]
        cx, cy = hand["center"]
        lmList = hand["lmList"]  # List of 21 landmark points
        fingers = detectorHand.fingersUp(hand)  # List of which fingers are up

        # Map values for easy drawing
        xVal = int(np.interp(lmList[8][0], [0, width], [0, width]))
        yVal = int(np.interp(lmList[8][1], [0, height], [0, height]))
        indexFinger = xVal, yVal

        if cy <= gestureThreshold:
            if fingers == [1, 0, 0, 0, 0]:
                print("Left")
                buttonPressed = True
                if imgNumber > 0:
                    imgNumber -= 1
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False
            if fingers == [0, 0, 0, 0, 1]:
                print("Right")
                buttonPressed = True
                if imgNumber < len(pathImages) - 1:
                    imgNumber += 1
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False

        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)  # Blue

        if fingers == [0, 1, 0, 0, 0]:
            if annotationStart is False:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            print(annotationNumber)
            annotations[annotationNumber].append(indexFinger)
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)  # Blue
        else:
            annotationStart = False

        if fingers == [0, 1, 1, 1, 0]:
            if annotations:
                annotations.pop(-1)
                annotationNumber -= 1
                buttonPressed = True
    else:
        annotationStart = False

    if buttonPressed:
        counter += 1
        if counter > delay:
            counter = 0
            buttonPressed = False

    for i, annotation in enumerate(annotations):
        for j in range(len(annotation)):
            if j != 0:
                cv2.line(imgCurrent, annotation[j - 1], annotation[j], (0, 0, 200), 12) # Light Blue

    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w - ws: w] = imgSmall

    cv2.imshow("Slides", imgCurrent)
    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break  # Quit

# Release resources
cap.release()
cv2.destroyAllWindows()
