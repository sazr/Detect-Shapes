# Prompt the user to draw a shape and try to detect what shape (book, tree, shirt, etc.)
# they drew. Zernike moments are used as a feature vector to describe the training set 
# and the users shape.

import os
import cv2
import glob
import mahotas
import imutils
import numpy as np 
from imutils import contours
from scipy.spatial import distance as dist

DEBUG       = True
ST_IDLE     = 0
ST_DRAWING  = 1
state       = ST_IDLE
pen_strokes = []
fvs         = []
fvs_labels  = []

def train():
	# From a list of images, get the Zernike moments as a feature vector and store in a list
	# We will determine what oject the user is drawing by their shape
	global fvs, fvs_labels

	# Calc zernike moments for each shape
	for img in glob.glob('./train/*.png'):
		src = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
		_, thresh = cv2.threshold(src, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		cnt = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1][0]

		# if DEBUG:
		# 	cv2.imshow('Shape', thresh)
		# 	cv2.waitKey(0)

		fv = mahotas.features.zernike_moments(thresh, cv2.minEnclosingCircle(cnt)[1], degree=8)
		fvs.append(fv)
		fvs_labels.append(os.path.splitext(os.path.basename(img))[0])

def handle_draw(frame):
	# Detect blue pen, store centroid point of blue pen, draw previous pen centroids
	global pen_strokes
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, (100, 157, 100), (130, 197, 255))
	mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))

	if DEBUG:
		cv2.imshow('mask', mask)

	cnts = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]

	if len(cnts) <= 0:
		return

	# get 5 largest contours and sort top to bottom
	cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)[:5]
	(cnts, boundingBoxes) = contours.sort_contours(cnts, method='top-to-bottom')
	pen = cnts[0]

	m = cv2.moments(pen)
	centroid = (int(m['m10'] / (m['m00'] + 1e-5)), int(m['m01'] / (m['m00'] + 1e-5)))
	pen_strokes.append( centroid )

	for i in range(len(pen_strokes)-1):
		cv2.line(frame, pen_strokes[i], pen_strokes[i+1], (0,255,0), 2)

	cv2.imshow('pen', frame)

def handle_idle(frame):
	# If we have pen strokes; attempt to detect the drawn object else do nothing
	global pen_strokes
	if len(pen_strokes) <= 0:
		return

	pen_strokes.append( pen_strokes[0] ) # close contour
	# pen_strokes = cv2.convexHull(pen_strokes, False)
	mask = np.zeros((frame.shape[0], frame.shape[1], 1), dtype='uint8')

	for i in range(len(pen_strokes)-1):
		cv2.line(mask, pen_strokes[i], pen_strokes[i+1], (255), 1)

	# cv2.drawContours(mask, [pen_strokes], -1, (255,), -1)
	cnt = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1][0]
	# Contour must be wide enough for mahotas zernike moments
	mask = imutils.resize(mask, width=500)

	if DEBUG:
		cv2.imshow('shape', mask)

	# Determine shape from zernike moments
	fv = mahotas.features.zernike_moments(mask, cv2.minEnclosingCircle(np.array(pen_strokes))[1], degree=8)
	D = dist.cdist([fv], fvs)
	i = np.argmin(D)

	if i >= 0 and i < len(fvs_labels):
		label = fvs_labels[i]
		print('Label: {}'.format(label))
	else:
		print('No shape detected')

	pen_strokes = []

def main():
	global state, pen_strokes

	cap = cv2.VideoCapture(0)

	train()	

	while True:
		res, frame = cap.read()

		if not res:
			break

		frame = imutils.resize(frame, height=300)
		cv2.imshow('frame', frame)

		if state == ST_DRAWING:
			handle_draw(frame)
		else:
			handle_idle(frame)

		key = cv2.waitKey(1)
		# Press escape to quit
		if key == 27:
			break
		# Press enter/return to change state from drawing to shape detection
		elif key == 13:
			state = ST_DRAWING if state == ST_IDLE else ST_IDLE

	cap.release()

if __name__ == "__main__":
	main()