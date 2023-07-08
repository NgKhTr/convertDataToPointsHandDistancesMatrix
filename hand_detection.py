from cvzone.HandTrackingModule import HandDetector
detector = HandDetector(detectionCon=0.8, maxHands=1)
def get_hand(img):
    # hands, img = detector.findHands(img)  # with draw
    hands = detector.findHands(img, draw=False)  # without draw
    bbox1 = None
    handType = None
    if hands:
        hand1 = hands[0]
        bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
        handType = hand1["type"]  # Handtype Left or Right
    return bbox1, handType