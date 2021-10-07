import cv2
import time
import os
import HandTrackingModule as htm

# Open webcam
wCam, hCam = 640, 480
capture = cv2.VideoCapture(0)
capture.set(3, wCam)
capture.set(4, hCam)

# Your images folder here
folderPath = "Fingerimages"
myList = os.listdir(folderPath)
print(myList)
overlayList = []

# Loop every images on folder above.
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
pTime = 0

# Instantiate object handTracking Module
detector = htm.handDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]
while True:
    success, img = capture.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    print(lmList)

    if len(lmList) != 0:
        fingers = []

        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)
        print(totalFingers)
        h, w, c = overlayList[totalFingers - 1].shape
        cv2.rectangle(img, (5, 5), (170, 200), (255, 255, 255), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 150), cv2.FONT_HERSHEY_PLAIN,
                    10, (0, 0, 255), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 0, 255), 3)
    cv2.imshow("Webcam", img)
    cv2.waitKey(1)


