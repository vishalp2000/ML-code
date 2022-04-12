'''
Authors: Joel, Tim, Daniel, Vishal
Date:    4/1/2022
Project: Kawaii-saki Robot Senior Project
Purpose: Collection of methods for analyzing input frames
'''
# Package imports
import math 
import cv2 as cv 
import numpy as np

# File imports
from payload import Payload
import messaging

# Global variables
resX = 640          # The x-component of resolution
resY = 480          # The y-component of resolution
offX = 0            # The x-component of camera offset from center of end-effector
offY = 0            # The y-component of camera offset from center of end-effector
my_categories = 2   # The number of types of lids
mm_per_pix = 2.958  # The amount of millimeters per pixel. This value is height-dependant
my_labels = ["Orange", "Yellow"]

# The array of minimum and maximum values for meaningful contours. Dimensions are in pixels
#             Orange  Yellow
my_target = [[85,     115],     # Min breadth
             [125,    155],     # Max breadth
             [85,     165],     # Min length
             [125,    205],     # Max length
             [8500,   23000],   # Min area
             [13500,  27000]]   # Max area

wide_net = [[1,      1],   
            [500,    500],
            [1,      1],
            [500,    500],
            [1,      1],
            [100000, 100000]]

def perspective(img):
    input_pts = np.float32([[634, 0], [636, 482], [28, 484], [9, 27]])
    output_pts = np.float32([[0, 0], [0, 488], [648, 488], [648, 0]])
    M = cv.getPerspectiveTransform(input_pts, output_pts)
    img = cv.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=(cv.INTER_LINEAR))
    return img

# Returns a list of 'Payload' types for each relevant contour in the frame
def getPayloads(img):
    # Preprocess the image
    image = cv.resize(img, (resX, resY), cv.INTER_AREA) # Conform the image to the desired resolution
    lower_blue = np.array([40, 30, 30])                 # Set the lower color bound                            
    upper_blue = np.array([170, 170, 190])              # Set the higher color bound
    image = cv.inRange(image, lower_blue, upper_blue)   # Conform every pixel to one or the other
    image = cv.GaussianBlur(image, (9, 9), 4)           # Blur the image to remove minor lines

    # Create the image for finding contours out of the preprocessed image
    thresh, cont = cv.threshold(image, 70, 255, cv.THRESH_BINARY_INV)            # Make every pixel white or black
    contours, thresh = cv.findContours(cont, cv.RETR_TREE, cv.CHAIN_APPROX_NONE) # Find all contours in the frame

    # Show the black and white image used for finding contours
    cv.imshow('', cont)

    # Populate a list of 'Payload' types from the contours
    cof = [int(resX/2 + offX), # The x-coord of the CoF
           int(resY/2 + offY)] # The y-coord of the CoF
    payloads = []              # The list of payloads is initially empty
    for contour in contours:
        # Draw the box around the contour
        my_bounds = cv.minAreaRect(contour)
        my_box = cv.boxPoints(my_bounds)
        my_box = np.int0(my_box)

        if bounds_just_right(my_bounds, my_target, my_categories):
            # Determine the center of the box
            M = cv.moments(my_box)
            cX = int(M['m10'] / M['m00']) # X-coord of box center in pixels
            cY = int(M['m01'] / M['m00']) # Y-coord of box center in pixels

            # Determine distances
            disX = cof[0] - cX # X-component of distance -- pos: right | neg: left
            disY = cof[1] - cY # Y-component of distance -- pos: down  | neg: up
            my_distance = np.sqrt(pow(disX, 2) + pow(disY, 2)) # Absolute distance

            new_payload = Payload()
            new_payload.bounds = my_bounds
            new_payload.x = pixels_to_mm(disX, mm_per_pix)
            new_payload.y = pixels_to_mm(disY, mm_per_pix)
            new_payload.distance = pixels_to_mm(my_distance, mm_per_pix)
            new_payload.pix_centroid = [cX, cY]
            new_payload.r = reference_rotation(my_bounds)
            new_payload.box = my_box
            new_payload.contour = contour
            new_payload.type = get_type(my_bounds, my_target, my_labels)
            payloads.append(new_payload)

    return payloads

# Converts a given distance in pixels to millimeters
def pixels_to_mm(pix_length, conversion):
    mm_length = pix_length * conversion
    return mm_length

def reference_rotation(bounds):
    adjust = 0
    breadth = bounds[1][0]
    length = bounds[1][1]
    if breadth < length:
        return 90 - bounds[2] + adjust
    return -bounds[2] + adjust

def adjust_sample_center(center, shape, sample_size):
    if center[0] < sample_size:
        adjust = sample_size - center[0]
        center = (center[0] + adjust, center[1])
    else:
        if center[0] > shape[1] - sample_size:
            adjust = sample_size - (shape[1] - center[0])
            center = (center[0] - adjust, center[1])
        elif center[1] < sample_size:
            adjust = sample_size - center[1]
            center = (center[0], center[1] + adjust)
        else:
            if center[1] > shape[0] - sample_size:
                adjust = sample_size - (shape[0] - center[1])
                center = (center[0], center[1] - adjust)
        return center

# Used by 'getPayloads' to reject extraneous contours (contours that are not lids)
def bounds_just_right(bounds, target, categories):
    if len(target[0]) != categories or len(target) != 6:
        print("Check 'target' parameter")
        return False
    i = 0
    while i < categories:
        if target[0][i] > target[1][i]:
            hold = target[0][i]
            target[1][i] = target[0][i]
            target[1][i] = hold
        if target[2][i] > target[3][i]:
            hold = target[2][i]
            target[3][i] = target[2][i]
            target[3][i] = hold

        breadth = bounds[1][0]
        length = bounds[1][1]
        area = breadth * length

        if breadth > length:
            hold = breadth
            breadth = length
            length = hold

        if (breadth >= target[0][i] and 
            breadth <= target[1][i] and 
            length >= target[2][i] and 
            length <= target[3][i] and 
            area >= target[4][i] and 
            area <= target[5][i]):
            return True
        i += 1
    return False

def get_type(bounds, target, labels):
    i = 0
    for color in labels:
        length = bounds[1][0]
        breadth = bounds[1][1]
        area = length * breadth
        if(area >= target[4][i] and area <= target[5][i]):
            return labels[i]
        i += 1
    return "None"

def sort_by_distance(payloads):
    i = 1
    payloads.sort(key = get_payload_distance)
    for payload in payloads:
        print(i, ": ", round(payload.distance, 2), " Type: ", payload.type)
        i += 1
    return payloads

def get_payload_distance(payload):
    return payload.distance

# Draws relevant contour information on the frame
def draw_payloads(img, payloads):
    for payload in payloads:
        if payload.bounds is not None:
            cX = payload.pix_centroid[0]
            cY = payload.pix_centroid[1]
            cv.drawContours(img, [payload.box], -1, (255, 0, 0), 2)
            cv.circle(img, (cX, cY), 5, (255, 255, 255), -1)
            midX = int(resX / 2 + offX)
            midY = int(resY / 2 + offY)
            cv.line(img, (midX, midY), (cX, cY), (0, 255, 0), 1)

            # Print the distance of the centroid to the CoF
            cv.putText(img,                             # Frame
                       str(round(payload.distance, 2)), # Distance in millimeters
                       (cX, cY),                        # Location on the frame
                       cv.FONT_HERSHEY_SIMPLEX,         # Font
                       0.5,                             # Font scale
                       (255, 255, 255),                 # Font color (BGR)
                       2)                               # Font thickness

            # Print the stats of the contour
            breadth = round(payload.bounds[1][0], 2)
            length =  round(payload.bounds[1][1], 2)
            area =    round(breadth * length, 2)
            angle = round(payload.bounds[2],2)
            #cv.putText(img, str("Len; " + str(length)),  (cX, cY+15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            #cv.putText(img, str("Bre: " + str(breadth)), (cX, cY+30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) 
            cv.putText(img, str("Area:  " + str(area)),   (cX, cY+15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if area > my_target[4][1]:
                cv.putText(img, str("Angle: " + str(angle)),  (cX, cY+30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv.putText(img, payload.type,  (cX, cY-50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
