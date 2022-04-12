# uncompyle6 version 3.8.0
# Python bytecode 3.7.0 (3394)
# Decompiled from: Python 3.8.0 (tags/v3.8.0:fa919fd, Oct 14 2019, 19:37:50) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: /home/pi/Desktop/graspf/segmentation.py
# Compiled at: 2022-04-12 11:51:52
# Size of source mod 2**32: 9278 bytes
import math, cv2 as cv, numpy as np
from payload import Payload
resX = 640
resY = 480
my_categories = 2
mm_per_pix = 3.124
my_target = [
 [
  80, 105],
 [
  110, 135],
 [
  80, 150],
 [
  110, 180],
 [
  7000, 18500],
 [
  11000, 21500]]

def perspective(img):
    input_pts = np.float32([[634, 0], [636, 482], [28, 484], [9, 27]])
    output_pts = np.float32([[0, 0], [0, 488], [648, 488], [648, 0]])
    M = cv.getPerspectiveTransform(input_pts, output_pts)
    img = cv.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=(cv.INTER_LINEAR))
    return img


def getPayloads(img):
    image = cv.resize(img, (resX, resY), cv.INTER_AREA)
    lower_blue = np.array([40, 30, 30])
    upper_blue = np.array([170, 170, 190])
    image = cv.inRange(image, lower_blue, upper_blue)
    image = cv.GaussianBlur(image, (9, 9), 4)
    thresh, cont = cv.threshold(image, 70, 255, cv.THRESH_BINARY_INV)
    contours, thresh = cv.findContours(cont, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cv.imshow('', cont)
    center = (0, 0)
    selected = -1
    payloads = []
    index = 0
    for contour in contours:
        my_bounds = cv.minAreaRect(contour)
        box = cv.boxPoints(my_bounds)
        box = np.int0(box)
        new_center = np.int0(my_bounds[0])
        nc_dx = abs(new_center[0] - image.shape[1] / 2)
        nc_dy = abs(new_center[1] - image.shape[0] / 2)
        new_distance = math.sqrt(nc_dx ** 2 + nc_dy ** 2)
        oc_dx = abs(center[0] - image.shape[1] / 2)
        oc_dy = abs(center[1] - image.shape[0] / 2)
        old_distance = math.sqrt(oc_dx ** 2 + oc_dy ** 2)
        if bounds_just_right(my_bounds, my_target, my_categories):
            new_payload = Payload()
            new_payload.bounds = my_bounds
            new_payload.x = int(new_center[0] * 6.481)
            new_payload.y = int(new_center[1] * 6.557)
            new_payload.r = reference_rotation(my_bounds)
            new_payload.distance = new_distance * 6.5
            new_payload.box = box
            new_payload.contour = contour
            new_payload.pix_centroid = [nc_dx, nc_dy]
            payloads.append(new_payload)
            if new_distance < old_distance:
                selected = index
                center = new_center
            index += 1

    if len(payloads) > 0:
        payloads[selected].selected = 1
    return payloads


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
        if breadth >= target[0][i] and breadth <= target[1][i] and length >= target[2][i] and length <= target[3][i] and area >= target[4][i]:
            if area <= target[5][i]:
                return True
        i += 1

    return False


def draw_payloads(img, payloads):
    i = 0
    for payload in payloads:
        if payloads[i].bounds is not None:
            cv.drawContours(img, [payloads[i].box], -1, (255, 0, 0), 2)
            M = cv.moments(payloads[i].box)
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            cv.circle(img, (cX, cY), 5, (255, 255, 255), -1)
            offX = 0
            offY = 0
            midX = int(resX / 2 + offX)
            midY = int(resY / 2 + offY)
            distance = np.sqrt(pow(midX - cX, 2) + pow(midY - cY, 2))
            cv.line(img, (midX, midY), (cX, cY), (0, 255, 0), 1)
            cv.putText(img, str(round(distance, 2)), (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,
                                                                                              255,
                                                                                              255), 2)
        i += 1