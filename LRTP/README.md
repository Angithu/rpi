# PJ_ccd_Contorller-

# performance
use base64 lib provide higher fps
for publish, fps keep 30 at least, range 30 to 75
for subscribe, fps keep 15 at least, range 15 to 60
if no use any lib, fps in range 1 to 7.

# publish
get frame by camera and encode with base64 lib
--------------------------------------------------------------------------
_, buffer = cv2.imencode('.jpg', frame) # Encoding the Frame		
jpg_as_text = base64.b64encode(buffer) # Converting into encoded bytes
--------------------------------------------------------------------------

then, publish the frame byte.
---------------------------------------------
client.publish(MQTT_TopicName, jpg_as_text)
---------------------------------------------

# subscribe
receive the frame byte and decode the byte with base64 lib
--------------------------------------------------------------------------------------
img = base64.b64decode(msg.payload)    
npimg = np.frombuffer(img, dtype=np.uint8) # converting into numpy array from buffer    
frame = cv2.imdecode(npimg, 1) # Decode to Original Frame
--------------------------------------------------------------------------------------

then, use cv2.imwrite write to image file.
----------------------------------
cv2.imwrite("output.jpg", frame)
----------------------------------
# PJ_ccd_Contorller
