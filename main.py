import cv2
import numpy as np
from generation_model import create_model
from preprocess_image import preprocess
from cvzone.HandTrackingModule import HandDetector

IMG_SIZE = 96
model = create_model()

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)

while True:
    # Get image frame
    success, img = cap.read()
    img = cv2.flip(img, 1)
    # Find the hand and its landmarks
    # hands, img = detector.findHands(img)  # with draw
    hands = detector.findHands(img, draw=False)  # without draw
    crop = img[:IMG_SIZE, :IMG_SIZE]
    if hands:
        # Hand 1
        hand1 = hands[0]
        # lmList1 = hand1["lmList"]  # List of 21 Landmark points
        bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
        cx,cy = hand1['center']  # center of the hand cx,cy
        # handType = hand1["type"]  # Handtype Left or Right
        # fingers1 = detector.fingersUp(hand1)
        x,y,w,h = bbox1
        a = max(w, h)//2 + 20
        print((cx-a),(cx+a),(cy-a),(cy+a))
        crop = img[(cy-a):(cy+a),(cx-a):(cx+a)]
        crop = preprocess(crop)
        crop = cv2.copyMakeBorder(crop, 50, 50, 50, 50, cv2.BORDER_CONSTANT, None, value=0)
        crop = cv2.resize(crop, (IMG_SIZE,IMG_SIZE))
        # X_predict = np.array([crop])
        # predict = str(np.argmax(model.predict([X_predict])))
        # img = cv2.putText(img, predict , (x + 3, y + 3) , cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display
    cv2.imshow("Image", img)
    cv2.imshow("preprocess", crop)
    # Press key "e" to exit
    if(cv2.waitKey(1) == ord('e')):
        break