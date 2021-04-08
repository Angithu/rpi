# -*- coding: utf-8 -*-

# Add path to python-common/TIS.py to the import path
# ~ import sys
# ~ sys.path.append("../python-common")

import cv2
import numpy as np
import os
import TIS
import time
import datetime
import configparser
from collections import namedtuple
import math
from scipy import ndimage
import matplotlib.pyplot as plt
#import base64
#import paho.mqtt.client as mqtt
import modbus_tk
import modbus_tk.defines as cst
import modbus_tk.modbus as modbus
import modbus_tk.modbus_tcp as modbus_tcp
import RPi.GPIO as GPIO
import time
import threading
input_value = 2
index = True
result = None
server = modbus_tcp.TcpServer(address = "127.0.0.1", port = 502) # Default port = 502
slaver = server.add_slave(1) # Slave_ID = 1

# This sample shows, how to get an image in a callback and use trigger or software trigger
# needed packages:
# pyhton-opencv
# pyhton-gst-1.0
# tiscamera

''' config '''
config = configparser.ConfigParser()
config.read('config.ini')

# camera config
light_IO            = config['Camera']['light_IO']
exposure            = config['Camera']['exposure']
exposure_auto       = config['Camera']['exposure_auto']
gain                = config['Camera']['gain']
gain_auto           = config['Camera']['gain_auto']
whitebalance_auto   = config['Camera']['whitebalance_auto']
whitebalance_red    = config['Camera']['whitebalance_red']
whitebalance_green  = config['Camera']['whitebalance_green']
whitebalance_blue   = config['Camera']['whitebalance_blue']
flash_time          = config['Camera']['flash_time']
signalpin           = config['GPIO']['signalpin']
ctrlpin             = config['GPIO']['ctrlpin']

# mqtt configure
#MQTT_ServerIP   = config['General']['Datalake_IP']
#MQTT_ServerPort = 1883 # int(config['General']['MQTT_ServerPort'])
#MQTT_TopicName  = config['Camera']['topic']

#client = mqtt.Client("client_camera")
#client.connect(MQTT_ServerIP, MQTT_ServerPort)

class CustomData:
    ''' Example class for user data passed to the on new image callback function '''    
    def __init__(self, newImageReceived, image):
        self.newImageReceived = newImageReceived
        self.image = image
        self.busy = False

CD = CustomData(False, None)

def on_new_image(tis, userdata):
    '''
    Callback function, which will be called by the TIS class
    :param tis: the camera TIS class, that calls this callback
    :param userdata: This is a class with user data, filled by this call.
    :return:
    '''
    # Avoid being called, while the callback is busy
    if userdata.busy is True:
        return

    userdata.busy = True
    userdata.newImageReceived = True
    userdata.image = tis.Get_image()
    userdata.busy = False

Tis = TIS.TIS()

# The following line opens and configures the video capture device.
# Tis.openDevice("41910044", 640, 480, "30/1",TIS.SinkFormats.BGRA, True)

# The next line is for selecting a device, video format and frame rate.
if not Tis.selectDevice():
    print("11111")
    quit(0)

#Tis.List_Properties()
Tis.Set_Image_Callback(on_new_image, CD)

# Tis.Set_Property("Trigger Mode", "Off") # Use this line for GigE cameras
Tis.Set_Property("Trigger Mode", False)
CD.busy = True # Avoid, that we handle image, while we are in the pipeline start phase
# Start the pipeline
Tis.Start_pipeline()

# Tis.Set_Property("Trigger Mode", "On") # Use this line for GigE cameras
Tis.Set_Property("Trigger Mode", True)
cv2.waitKey(1000)

CD.busy = False  # Now the callback function does something on a trigger

# Remove comment below in oder to get a propety list.
# ~ Tis.List_Properties()

# In case a color camera is used, the white balance automatic must be
# disabled, because this does not work good in trigger mode
Tis.Set_Property("Whitebalance Auto", bool(whitebalance_auto))
# ~ Tis.Set_Property("Whitebalance Red", int(whitebalance_red))
# ~ Tis.Set_Property("Whitebalance Green", int(whitebalance_green))
# ~ Tis.Set_Property("Whitebalance Blue", int(whitebalance_blue))

# Gain Auto
# Check, whether gain auto is enabled. If so, disable it.
if(gain_auto=="False"):
    if Tis.Get_Property("Gain Auto").value: Tis.Set_Property("Gain Auto", False) # bool(gain_auto)
    
# Gain
Tis.Set_Property("Gain", int(gain))


# exposure Auto
# Now do the same with exposure. Disable automatic if it was enabled
# then set an exposure time.
if(exposure_auto=="False"):
    if Tis.Get_Property("Exposure Auto").value: Tis.Set_Property("Exposure Auto", False) # bool(exposure_auto)
        
# exposure
Tis.Set_Property("Exposure Time (us)", int(exposure)) # 24000

print("\n===========================")
print("Light IO : {0}".format(light_IO))
print("Gain Auto : %s " % Tis.Get_Property("Gain Auto").value)
print("Gain : %d" % Tis.Get_Property("Gain").value)
print("Exposure Auto : %s " % Tis.Get_Property("Exposure Auto").value)
print("Exposure(us) : %d" % Tis.Get_Property("Exposure Time (us)").value)
print("Flash Time : {0}".format(flash_time))
print("Whitebalance Auto : {0}".format(whitebalance_auto))
print("Whitebalance Red : {0}".format(whitebalance_red))
print("Whitebalance Green : {0}".format(whitebalance_green))
print("Whitebalance Blue : {0}".format(whitebalance_blue))

# ~ print("\n===========================")
# ~ print("Data lake IP: {}".format(MQTT_ServerIP))
# ~ print("MQTT ServerPort: {}".format(MQTT_ServerPort))
# ~ print("Topic: {}".format(MQTT_TopicName))

trigger = False
error = 0

# ~ lastkey = 0
print('\n===========================\nPress Esc to stop')

''' light '''
def light():
    global trigger, index, input_value
    global light_IO
    global flash_time
    
    cur_th = threading.currentThread()
    
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(20,GPIO.OUT)
    #GPIO.setup(16,GPIO.OUT, initial =GPIO.LOW)
    GPIO.setup(5,GPIO.IN) 
    
    while getattr(cur_th, "do_light", True): 
        input_value = GPIO.input(5)
        #print(input_value)
        #print(index)
        #print(trigger)
        if(input_value == 1 and index == True and trigger == True):
            GPIO.output(20,True)
            print("lightopen"+datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S-%f'))
            time.sleep(float(flash_time))#flash_time
            GPIO.output(20,False)
            print("lightclose"+datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S-%f'))
            #time.sleep(0.5)
            #index = False
            trigger = False
            
        elif(input_value == 0 and index == False):
            index = True
    GPIO.cleanup()
    print("GPIO clean..!")

def algorithm(img):
    global result
    """ config """
    flg_slope = False
    angel = 21.9

    """ image source """
    img = cv2.imread('/home/pi/alien/PJ_LRTP-master/img/sample/191116_101744_0000000083_CAM1_OK.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


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
    #plt.subplot(131)
    #plt.imshow(I1)
    #plt.show()

    """ equalization """
    uni_Ieq = cv2.equalizeHist(I1[:,:,2])
    #plt.subplot(132)
    #plt.imshow(uni_Ieq, cmap = 'gray')
    #plt.show()

    """ blur """
    blur = cv2.GaussianBlur(uni_Ieq, (5, 5), 0)
    #plt.subplot(133)
    #plt.imshow(blur, cmap = 'gray')
    #plt.show()

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
    # SET VALUE
    slaver.set_values("A", 0, result)
    slaver.set_values("A", 1, result)
    print("result: {0}".format(result))

def camera():
    global index,input_value,trigger
    #cam_th = threading.currentThread()
    try:
        while True:
                
            start = time.time()
            
            ''' trigger by time '''
            trigger = True
            time.sleep(float(0.2))
            
            Tis.Set_Property("Software Trigger",1) # Send a software trigger
                        
            # Wait for a new image. Use 10 tries.
            tries = 10
            while CD.newImageReceived is False and tries > 0:
                time.sleep(0.0005)
                tries -= 1
                #print(tries)
            #print(CD.newImageReceived)
            # Check, whether there is a new image and handle it.
            #print("1:CD  " + str(CD.newImageReceived))
            #trigger = True
            if CD.newImageReceived is True:
                CD.newImageReceived = False
                #input_value = GPIO.input(20)
                #print("2:  " + str(input_value))
                #print("3:  " + str(CD.newImageReceived))
                #input_value = GPIO.input(20)
                #trigger = True
                if(input_value == 1 and index == True):
                    trigger = True
                    #time.sleep(0.09)
                    print("camera"+datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S-%f'))
                    
                               
                # base64 encode		
                #_, buffer = cv2.imencode('.jpg', CD.image) # Encoding the Frame		
                #jpg_as_text = base64.b64encode(buffer) # Converting into encoded bytes
                
                # publish
                #client.publish(MQTT_TopicName, jpg_as_text)
                
                # image processing
                    
                    #algorithm(CD.image)
                    index = False
                    print("algorithm done")
                    #print(datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S'))
                    # save image
                    #fileName = "/home/pi/alien/an/LRTP/image/{0}.jpg".format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
                    fileName = "/home/pi/alien/an/LRTP/image/{0}.jpg".format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
                    cv2.imwrite(fileName, CD.image)
                    
                    print("cam done"+datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S-%f'))
                    
                    #cv2.waitKey(1)
            # ~ else:
                # ~ print("No image received")
                    
                cv2.waitKey(1)  
                trigger = False
            
                # fps
            end = time.time()
            t = end - start
            fps = 1/t
    # ~ print("{} capture image, process time: {:3} ms, fps: {:2}, ".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), int(t*1000), int(fps)))

        #time.sleep(1)
                
    except KeyboardInterrupt:
            print("\n")

            # Stop the thread of light
            if(light_IO): th_light.do_light = False
            #th_camera.do_camera = False
            # Stop the pipeline and clean ip
            Tis.Stop_pipeline()
            print("Tis clean..!")
              
            destory()
            # mqtt
            #client.disconnect()
            print("Data lake clean..!")      

            print('Program ends')


def modbus_setup():
    slaver.add_block("A", cst.HOLDING_REGISTERS, 0, 16)
    # slaver.set_values("A", 0, 0)  
    
def loop():
    global result
    # START
    server.start()    
    while True:
        # SET VALUE
        slaver.set_values("A", 0, result)
        slaver.set_values("A", 1, result)
        
        # DELAY
        time.sleep(1)
        
def destory():
    # STOP
    server.stop()
    
def key():
    try:
        while True:
            a=0
    except KeyboardInterrupt:
        print("\n")

        # Stop the thread of light
        if(light_IO): th_light.do_light = False
        #th_camera.do_camera = False
        # Stop the pipeline and clean ip
        Tis.Stop_pipeline()
        print("Tis clean..!")
          
        destory()
        # mqtt
        #client.disconnect()
        print("Data lake clean..!")      

        print('Program ends')


if __name__=='__main__':
    modbus_setup()
    if(light_IO):
        th_light = threading.Thread(target = light)
        th_light.start()
    else:
        print("Flash Light Close..(light_IO={0})".format(light_IO))	
    #th_modbus = threading.Thread(target = loop)
    #th_modbus.start()
    server.start()
    camera()
