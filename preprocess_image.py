import cv2
import numpy as np

def preprocess(img):
	img_root = img[5:-5, 5:-5]
	lab = cv2.cvtColor(img_root, cv2.COLOR_BGR2Lab)
	l_channel, a, b = cv2.split(lab)
	clade = cv2.createCLAHE(clipLimit=2., tileGridSize=(11,11))
	cl = clade.apply(l_channel)
	limg = cv2.merge((cl,a,b))
	img = cv2.cvtColor(limg, cv2.COLOR_BGR2GRAY)

	# Max constrast
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
	topHat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel, iterations = 55)
	blackHat =cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel, iterations = 55)
	imgGrayscalePlusTopHat = cv2.add(img, topHat)
	imgMaxConstrast = cv2.subtract(imgGrayscalePlusTopHat, blackHat)
	# imgBlur = cv2.GaussianBlur(imgMaxConstrast, (3,3), 0)
	_, img_gray = cv2.threshold(imgMaxConstrast, 20, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	return img_gray