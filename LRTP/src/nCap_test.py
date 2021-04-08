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

import base64
import paho.mqtt.client as mqtt

import RPi.GPIO as GPIO
import time
import threading


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
MQTT_ServerIP   = config['General']['Datalake_IP']
MQTT_ServerPort = 1883 # int(config['General']['MQTT_ServerPort'])
MQTT_TopicName  = config['Camera']['topic']

client = mqtt.Client("client_camera")
client.connect(MQTT_ServerIP, MQTT_ServerPort)

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

print("\n===========================")
print("Data lake IP: {}".format(MQTT_ServerIP))
# ~ print("MQTT ServerPort: {}".format(MQTT_ServerPort))
print("Topic: {}".format(MQTT_TopicName))

trigger = False
error = 0
# ~ lastkey = 0
print('\n===========================\nPress Esc to stop')

''' light '''
def light():
    global trigger
    global light_IO
    global flash_time
    
    cur_th = threading.currentThread()
    
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(21,GPIO.OUT)
        
    while getattr(cur_th, "do_light", True):
        if(trigger):
            GPIO.output(21,True)
            time.sleep(float(flash_time))
            GPIO.output(21,False)
            
    GPIO.cleanup()
    print("GPIO clean..!")

if(light_IO):
    th_light = threading.Thread(target = light)
    th_light.start()
else:
    print("Flash Light Close..(light_IO={0})".format(light_IO))

try:
        
    while True:        
        ''' trigger by input '''
        '''
        trigger = False
        input_text = input("Press space + enter to trigger an image.\n q + enter to stop the stream.")
        if input_text == "q":
            break
        elif input_text == " ":  
            trigger = True
            time.sleep(0.05)
            
            Tis.Set_Property("Software Trigger",1) # Send a software trigger
                        
            # Wait for a new image. Use 10 tries.
            tries = 10
            while CD.newImageReceived is False and tries > 0:
                time.sleep(0.01)
                tries -= 1
                                
            # Check, whether there is a new image and handle it.
            if CD.newImageReceived is True:
                CD.newImageReceived = False
                
                fileName = "../img/{0}.jpg".format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
                cv2.imwrite(fileName, CD.image)
            else:
                print("No image received")
            
            print("Gain: {0}".format(Tis.Get_Property("Gain").value))
            print("exposure: {0}".format(Tis.Get_Property("Exposure Time (us)").value))
            
            lastkey = cv2.waitKey(10) 
            ''' 
            
        start = time.time()
        
        ''' trigger by time '''
        trigger = True
        time.sleep(float(flash_time))
        
        Tis.Set_Property("Software Trigger",1) # Send a software trigger
                    
        # Wait for a new image. Use 10 tries.
        tries = 10
        while CD.newImageReceived is False and tries > 0:
            time.sleep(0.0005)
            tries -= 1
                            
        # Check, whether there is a new image and handle it.
        if CD.newImageReceived is True:
            CD.newImageReceived = False
            
            # to byteArray
            # ~ picbyte = (CD.image).tobytes()
            
            # method1
            # ~ picbyte = bytes(CD.image)
            
            # method2
            # ~ picbyte = str(CD.image)
            
            # base64 encode		
            _, buffer = cv2.imencode('.jpg', CD.image) # Encoding the Frame		
            jpg_as_text = base64.b64encode(buffer) # Converting into encoded bytes
			
			# publish
            client.publish(MQTT_TopicName, jpg_as_text)
            
            fileName = "../img/{0}.jpg".format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
            cv2.imwrite(fileName, CD.image)
        else:
            print("No image received")
                
        cv2.waitKey(10)  
        trigger = False
        
        # fps
        end = time.time()
        t = end - start
        fps = 1/t
        print("{} capture image, process time: {:3} ms, fps: {:2}, ".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), int(t*1000), int(fps)))
        
        time.sleep(1)
            
            
except KeyboardInterrupt:
    print("\n")
    
    # Stop the thread of light
    if(light_IO): th_light.do_light = False
    
    # Stop the pipeline and clean ip
    Tis.Stop_pipeline()
    print("Tis clean..!")  
    
    # mqtt
    client.disconnect()
    print("Data lake clean..!")      
    
    print('Program ends')
