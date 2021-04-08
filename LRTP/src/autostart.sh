#!/bin/bash

cd /home/pi/alien/projectA/src
python3 node_cap.py

echo "waiting 10 seconds for program initialize..."
sleep 10

cd /home/pi/alien/projectA/src
python3 node_algoManager.py

echo "waiting 5 seconds for program initialize..."
sleep 5

cd /home/pi/alien/projectA/src
sudo python3 node_modbus.py

echo "program startup!"
