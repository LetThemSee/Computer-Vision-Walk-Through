#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 09:30:37 2018

@author: jinzhao
"""

import cv2
import numpy as np
from ps6_functions import *
import time

if __name__ == "__main__":
    video_path = 'input/pedestrians.avi'
    cap = cv2.VideoCapture(video_path)
    # ------------------------------------------------------------------------
    frame_init_rgb = get_frame(video_path, 0) 
    x_start, y_start, size_x, size_y = load_file('input/pedestrians.txt')
    model = frame_init_rgb[y_start:y_start+size_y-150, x_start:x_start+size_x, :].astype('float32')
    
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    sample_space = (0, 0, width, height)
    
    #out = cv2.VideoWriter("output_ped.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, (int(width), int(height)))
    tracker = ParticleTracker_MSL(model, sample_space)
    count = 0
    while(cap.isOpened()):
        ret, frame_rgb = cap.read()
        if not ret:
            break # unsuccessful reading 
        frame = np.float32(frame_rgb)
        
        tracker.update(frame)
 
        frame_draw = tracker.visualize_frame(frame_rgb)
        
        cv2.imshow('image', frame_draw)
        #out.write(frame_draw)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
        count += 1
        #if count == 28 or count == 84 or count == 144:
            #cv2.imwrite(str(count)+'_ped.png', frame_draw)
    
    cap.release()
    cv2.destroyAllWindows()
    