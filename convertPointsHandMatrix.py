import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import os

FOLDER_DATASET_PATH = 'data'
CLASSES = ["A", "B", "C"] # nhớ thêm dô, NHỚ BỎ nothing ra ngoài
CLASS_ID = {class_name: id for class_name, id in enumerate(CLASSES)}
FILE_DATA_OUT = "data.np"

detector = HandDetector(maxHands=1)

def convertPointsHandToDistanceMatrix(pointsHangMatric: np.ndarray):
    # distances = [np.linalg.norm(p1 - p2) for p1 in range]
    distances = []
    for i_p1 in range(pointsHangMatric.shape[0] - 1):
        for i_p2 in range(i_p1 + 1, pointsHangMatric.shape[0]):
            distances.append(np.linalg.norm(pointsHangMatric[i_p1] - pointsHangMatric[i_p2]))
    return np.array(distances)

def convertImageToPointsHand(filePath):
    img = cv2.imread(filePath)
    hand, _ = detector.findHands(img)

    # có thể có check điểu kiện ktra tay
    
    return hand['lmList']

def convertDataset():

    data = []
    for folderClass in os.listdir(FOLDER_DATASET_PATH):
        folderClassPath = os.path.join(FOLDER_DATASET_PATH, folderClass)
        if (not os.path.isdir(folderClassPath)) or (folderClass not in CLASS_ID):
            continue
        
        data_per_class = []
        for file in os.listdir(folderClassPath):
            filePath = os.path.join(folderClassPath, file)
            if os.path.isdir(filePath):
                continue
            distanceMatrix = convertPointsHandToDistanceMatrix(convertImageToPointsHand(filePath))
            data_per_class.append(distanceMatrix)

        data.append([data_per_class, CLASS_ID[folderClass]])

    return data


def main():
    if os.path.isdir(FOLDER_DATASET_PATH):
        data = convertDataset()
        np.save(data, FILE_DATA_OUT)
    else:
        print("Folder dataset không tồn tại")