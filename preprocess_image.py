import cv2
import os

def preprocess(img):
	# img = img[5:-5, 5:-5]
	lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
	l_channel, a, b = cv2.split(lab)
	clade = cv2.createCLAHE(clipLimit=2., tileGridSize=(11,11))
	cl = clade.apply(l_channel)
	limg = cv2.merge((cl,a,b))
	img = cv2.cvtColor(limg, cv2.COLOR_BGR2GRAY)
	img = cv2.GaussianBlur(img, (7,7), 0)

	# Max constrast
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13,13))
	topHat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel, iterations = 45)
	blackHat =cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel, iterations = 45)
	imgGrayscalePlusTopHat = cv2.add(img, topHat)
	imgMaxConstrast = cv2.subtract(imgGrayscalePlusTopHat, blackHat)
	_, img_gray = cv2.threshold(imgMaxConstrast, 20, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	return img_gray

# path_dir = "D:\project\sign_language_recognize/test_data\A/"
# for f in os.listdir(path_dir):
# 	img = cv2.imread(path_dir + f)
# 	img = preprocess(img)
# 	cv2.imshow(f, img)
# 	if(cv2.waitKey() == ord('n')):
# 		cv2.destroyAllWindows()	