import cv2 as cv
from AImodel import AIModel
from payload import Payload
from PIL import Image
import segment_frame as seg
from time import time as now
from time import sleep
#from messaging import packet
import messaging
import numpy as np
import threading
import multiprocessing
import math
from copy import deepcopy
from math import sqrt
import os
from picamera.array import PiRGBArray
from picamera import PiCamera

'''
x = numpy.zeros((3,15,20))
x1 = numpy.reshape(x[0], (1,300))
x2 = numpy.reshape(x[1], (1,300))
x3 = numpy.reshape(x[2], (1,300))
y = numpy.concatenate((x1, x2, x3), 1)
print(x.shape, y.shape)
exit()
'''

#video server
#threading.Thread(target=lambda: flask.main()).start()
global i
i = 0
# model = AIModel((1296,976), 440)
model = AIModel()
heights = [-254, -374, -343, -275, 0 ]#change with distance sensor

path = "/home/pi/Desktop/graspf"
template = cv.imread(os.path.join(path, 'test0.jpg'))
template = cv.resize(template,(648,488))
template = template[60:480,70:610]
template = cv.resize(template, (160,120))

scada = { 'robot_tags':{'home':True} }
prediction = -1 #4 to 2
zoom = 1.5
mm_per_pix = 3.124 # The number of mm equal to 1 pixel - must be measured and changes with height

# Initialize the PiCamera Object and frame
try:
    cam = PiCamera()
    cam.resolution = (640, 480)
    cam.framerate = 30
except:
    print("Camera Error")
raw_capture = PiRGBArray(cam, size = (640, 480))
sleep(0.1) # allow the camera to warm up

goaltype = "a" # Replace with socket reading
while not goaltype == "":
    for frame in cam.capture_continuous(raw_capture, format="bgr", use_video_port=True):
        # global conn
        start = now()
        img = frame.array

        raw_capture.truncate(0)

        payloads = seg.getPayloads(img)
        seg.sort_by_distance(payloads)
        
        bounding_box = None
        x = 0
        y = 0
        selected = 0
        for index, payload in enumerate(payloads):
            if payload.selected:
                selected = index
                # bounding_box, sample = seg.getPayload(payload, img)
                x = seg.pixels_to_mm(payload.x, mm_per_pix)
                y = seg.pixels_to_mm(payload.y, mm_per_pix)
        
        my_labels = ["Yellow", "Orange"]

        seg.draw_payloads(img, payloads)

        # tag_set = Payload().tags()

        final = cv.resize(img,(640, 480))

        # scada = messaging.client_send('vision', tag_set, True)
        #print(scada['scada_tags'])

        cv.imshow("Press 'q' to quit", final)
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        for payload in payloads:
            if payload.type == goaltype:
                xMove = payload.x
                yMove = payload.y
                rot = payload.r
                tag_set = {'right_robot_x':xMove, 'right_robot_y':yMove, 'right_robot_r':rot}
                node = messaging.client_send('vision', tag_set, True)
                print(node['vision_tags']['right_robot_x'])
                # Move to and pick up this payload

    goaltype = "a" # Reset flag
    cam.close()
    cv.destroyAllWindows()
