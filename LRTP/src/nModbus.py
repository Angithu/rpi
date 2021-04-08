
import time
import datetime
import logging
import threading
import configparser

import paho.mqtt.client as mqtt

import modbus_tk
import modbus_tk.defines as cst
import modbus_tk.modbus as modbus
import modbus_tk.modbus_tcp as modbus_tcp

# ~ logger = modbus_tk.utils.create_logger(name = "console", record_format = ">> %(message)s")

''' config '''
config = configparser.ConfigParser()
config.read('config.ini')

# mqtt configure
MQTT_ServerIP   = config['General']['Datalake_IP']
MQTT_ServerPort = 1883 # int(config['General']['MQTT_ServerPort'])
MQTT_SubTopic   = config['Algorithm']['topic']

# modbus
modbus_ip = config['Modbus']['ip']
modbus_port = 502 # int(config['Modbus']['port'])
modbus_slave = int(config['Modbus']['slaveID'])
modbus_dataCnt = int(config['Modbus']['data_cnt'])

# Create server
server = modbus_tcp.TcpServer(address = modbus_ip, port = modbus_port) # Default port = 502
slaver = server.add_slave(modbus_slave) # Slave_ID = 1

data = 0

print("===========================")
print("Data lake IP: {}".format(MQTT_ServerIP))
print("S_Topic: {}".format(MQTT_SubTopic))

print("\n===========================")
print("Modbus IP: {}".format(modbus_ip))
print("Modbus SlaveID: {}".format(modbus_slave))
print("Modbus Data Count: {}".format(modbus_dataCnt))

''' mqtt '''
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe(MQTT_SubTopic) # sub single topic
    
def on_message(client, userdata, msg):
    global data
    
    data = int(msg.payload.decode("utf-8"))
    # ~ print("{0} receive data: {1}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),msg.payload.decode("utf-8")))

''' modbus '''
def setup():
    slaver.add_block("A", cst.HOLDING_REGISTERS, 0, modbus_dataCnt)
       
def modbus():
    global data
    
    # ~ logger.info("running...")
    print("\n===========================")
    print("modbus startup..")
    cur_th = threading.currentThread()    
    
    # START
    server.start()    
    while getattr(cur_th, "do_modbus", True):
        start = time.time()
        
        # SET VALUE
        slaver.set_values("A", 0, data)
        
        # fps
        end = time.time()
        t = end - start
        fps = 1/t
        print("{} communication done, data: {:4}, procTime: {:3.2f} ms".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), data, (t*1000)))
            
        # DELAY
        time.sleep(0.5)
        
def destory():
    # ~ logger.info("destory")
    # STOP
    server.stop()
       
if __name__ == "__main__":
    setup()
    
    try:
        ''' modbus '''
        th_modbus = threading.Thread(target = modbus)
        th_modbus.start()
        
        ''' mqtt subscriber '''
        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(MQTT_ServerIP, MQTT_ServerPort, 60)
        client.loop_forever()
        
        # ~ loop()
    except KeyboardInterrupt:
        print("\n")
        
        th_modbus.do_modbus = False
        print("Thread modbus clean..!")   
        
        destory()
        print("modbus server stop..!")        
        
        print("Modbus END!")
