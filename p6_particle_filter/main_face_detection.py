#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 10:28:35 2018

@author: jinzhao
"""

import numpy as np
import cv2
from ps6_functions import *

face_cascade = cv2.CascadeClassifier('/Users/jinzhao/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

video_path = 'input/pres_debate.avi'
frame_init_rgb = get_frame(video_path, 0) # Gray image
x_start, y_start, size_x, size_y = load_file('input/pres_debate.txt')
frame_init = cv2.cvtColor(frame_init_rgb, cv2.COLOR_BGR2GRAY)
#img_show([frame_init])

faces = face_cascade.detectMultiScale(frame_init)
for (x, y, w, h) in faces:
    cv2.rectangle(frame_init_rgb, (x,y), (x+w,y+h), (255,0,0),2)
    roi_gray = frame_init[y:y+h, x:x+w]
    roi_color = frame_init_rgb[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray) #Find eyes in ROI(region of interest)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0),2)
        
cv2.imshow('img', frame_init_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()


