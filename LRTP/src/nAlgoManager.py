#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:27:39 2019

@author: pi
"""

import base64
import datetime
import time
import paho.mqtt.client as mqtt
import numpy as np
import cv2
import threading
import configparser

import algorithm
from file_monitor import FileMonitor


'''
receive image from mqtt subscriber
send to Algorithm to get result of algorithm
then, send the result out by modbus_tcp protocol
'''


''' config '''
config = configparser.ConfigParser()
config.read('config.ini')

# mqtt configure
MQTT_ServerIP   = config['General']['Datalake_IP']
MQTT_ServerPort = 1883 # int(config['General']['MQTT_ServerPort'])
MQTT_TopicName  = config['Algorithm']['topic']
MQTT_SubTopic   = config['Camera']['topic']

# image
img_res_w   = int(config['Camera']['resolution_w'])
img_res_h   = int(config['Camera']['resolution_h'])

flg_start = False
frame = np.zeros((img_res_h, img_res_w, 3), np.uint8)

client = mqtt.Client("client_algorithm")
client.connect(MQTT_ServerIP, MQTT_ServerPort)

print("===========================")
print("Data lake IP: {}".format(MQTT_ServerIP))
print("S_Topic: {}".format(MQTT_SubTopic))
print("P_Topic: {}".format(MQTT_TopicName))

''' hmr '''
# register file_monitor
fm = FileMonitor()
fm.add_file("algorithm.py", algorithm)
fm.start()
print("\n===========================")
print("FM ready..")

''' mqtt subscriber '''
def on_connect(client, userdata, flags, rc):
    print("\n===========================")
    print("Connected with result code " + str(rc))    
    client.subscribe(MQTT_SubTopic) # sub single topic

def on_message(client, userdata, msg):
    global frame
    global flg_start
        
    # Decoding the message
    img = base64.b64decode(msg.payload)    
    npimg = np.frombuffer(img, dtype = np.uint8) # converting into numpy array from buffer    
    frame = cv2.imdecode(npimg, 1) # Decode to Original Frame
    
    flg_start = True
    
''' algorithm manager '''
def subProcess():
    '''
    get the image and send to algorithm to get result, then publish by mqtt publisher
    '''
    global flg_start
    global frame
    global MQTT_TopicName
    
    cur_th = threading.currentThread()        
    while getattr(cur_th, "do_algorithm", True):
        if(flg_start):
            start = time.time()
            flg_start = False
            
            ins_algorithm = algorithm.Algorithm(frame)
            result = int(ins_algorithm.run()) # get result from algorithm
            
            # publish
            client.publish(MQTT_TopicName, result)
            
            # fps
            end = time.time()
            t = end - start
            fps = 1/t
            print("{} image process done, result: {:4}, procTime: {:4.2} ms".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), result, (t*1000)))
            
            
''' main '''
try:
    th_subProc = threading.Thread(target = subProcess)
    th_subProc.start()
    print("algorithm ready..")    
    
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_ServerIP, MQTT_ServerPort, 60)
    print("data lake ready..")     
    client.loop_forever()
    
except:
    print("\n")
    
    fm.stop_monitor()
    print("FM clean..!")
    
    th_subProc.do_algorithm = False
    print("algorithm clean..!")
    
    print("Algorithm Manager END!")
