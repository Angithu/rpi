# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:52:44 2020

@author: 00048766
"""

import cv2

cam = cv2.VideoCapture(0)
# ~ cam.set(15,32)

try:
	while True:
		ret, img = cam.read()
		# ~ cam.set(14,326)
		# ~ cam.set(15,32)
		
		vis = img.copy()
		cv2.imshow('Camera', vis)
		
		if(0xFF & cv2.waitKey(5)==27):
			break
except:
	print("\n")
	print(cam.get(15))
	cv2.destroyAllWindows()
