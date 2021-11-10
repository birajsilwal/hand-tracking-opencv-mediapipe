import cv2 as cv
import mediapipe as mp
import time

prevTime = 0
currTime = 0

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open camera")
    exit()

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while cap.isOpened():
    success, img = cap.read()
    # if frame is read correctly ret is True
    if not success:
        print("[ERROR] Can't receive frame (stream end?). Exiting ...")
        break

    # flip image horizontally
    img = cv.flip(img, 1)
    # hands object only takes RGB
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # detecting hands and drawing dots and lines
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)

    # calculating frame per second
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    # displaying fps
    cv.putText(img, 'FPS: ' + str(int(fps)), (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

    cv.imshow("Image", img)
    if cv.waitKey(1) == ord('q'):
        break
