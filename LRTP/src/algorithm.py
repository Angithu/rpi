# -*- coding: utf-8 -*-

"""
Created on Thu Jul  9 14:03:59 2020

@author: 00048766
"""

import cv2
import math
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

""" config """
flg_slope = False
angel = 21.9

""" image source """
img = cv2.imread('/home/pi/alien/PJ_LRTP-master/img/191116_101744_0000000083_CAM1_OK.jpg')
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


Io = np.zeros(img.shape, dtype = np.uint8)
Io[:,:,0] = img[:,:,0]
Io[:,:,1] = img[:,:,1]
Io[:,:,2] = img[:,:,2]

plt.imshow(Io)
plt.show()

""" segmentation """
Io = img[150:750, 400:900] #[h,w]

plt.imshow(Io)
plt.show()

""" calculate slope """
if(flg_slope):
    #tmp_l1 = np.zeros(Io[:,:,0][:,0].shape, dtype = np.uint8)
    #tmp_l2 = np.zeros(Io[:,:,0][:,0].shape, dtype = np.uint8)
    tmp_l = np.zeros(Io[:,:,0][:,0].shape, dtype = np.uint16)
    
    for i in range(600):
        tmp_l[i] = i+1
    
    ''' array slice
    [col, row] col-horizen; row-verticle
    [...,0] catch all row
    [0,...] catch all column 
    '''
    tmp_l1 = Io[...,0][:,0]   
    tmp_l2 = Io[...,0][:,499]
    
    ''' get point of slope edge '''
    for i in range(200,300):
        if(tmp_l1[i]<200):
            p1 = i; break
    for i in range(100):
        if(tmp_l2[i]<200):
            p2 = i; break
    
    ''' get angel '''
    angel = math.degrees(math.atan((p1 - p2)/(500)))
#    print("rotate angel: {0}".format(angel))

""" rotate """
im_rotate = ndimage.rotate(Io, 0-angel, reshape = False)
plt.imshow(im_rotate)
plt.show()

I1 = im_rotate[100:500, 200:300] #[h,w]
plt.subplot(131)
plt.imshow(I1)
#plt.show()

""" equalization """
uni_Ieq = cv2.equalizeHist(I1[:,:,2])
plt.subplot(132)
plt.imshow(uni_Ieq, cmap = 'gray')
#plt.show()

""" blur """
blur = cv2.GaussianBlur(uni_Ieq, (5, 5), 0)
plt.subplot(133)
plt.imshow(blur, cmap = 'gray')
plt.show()

plt.imshow(blur, cmap = 'gray')
plt.show()

""" get edge """
grayAry = []
maxSlope = 0
h, w = I1[:,:,2].shape
tmp_rx = np.zeros(I1[:,:,0][:,0].shape, dtype = np.uint16)
for i in range(400): # for plot coor_x
        tmp_rx[i] = i+1
        
for i in range(0, h): # edge1 - Top plate
    sumGray = 0    
    for j in range(w):
        sumGray += blur[i, j]
    
    grayAry = np.append(grayAry, sumGray)
    
    ''' get edge 1 - plate'''
    if(i>=40 and i<80):
        if((grayAry[i-1] - grayAry[i]) > maxSlope):
            maxSlope = (grayAry[i-1] - grayAry[i])
            edge_p1 = i
    
    ''' get edge 2 - wafer '''
    if(i==100): maxSlope = 0
    elif(i>=290 and i<330):
        if((grayAry[i] - grayAry[i-1]) > maxSlope):
            maxSlope = (grayAry[i] - grayAry[i-1])
            edge_p2 = i-1
    
plt.plot(tmp_rx, grayAry)
plt.show()

""" result """
result = edge_p2 - edge_p1
print("result: {0}".format(result))
