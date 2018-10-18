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
    start_time = time.time()
    
    video_path = 'input/pres_debate.avi'
    cap = cv2.VideoCapture(video_path)
    # 
    # ------------------------------------------------------------------------
    frame_init_rgb = get_frame(video_path, 0) # Gray image
    x_start, y_start, size_x, size_y = load_file('input/pres_debate_hand.txt')
    frame_draw = cv2.rectangle(frame_init_rgb, (x_start, y_start), (x_start+size_x, y_start+size_y), (0, 255, 0), 3) # UL corner -- DR corner
    #img_show([frame_draw])
    #cv2.imwrite('output/ps6-1-a-1.png', frame_draw[y_start:y_start+size_y, x_start:x_start+size_x])
    
    # Tracking
    # ------------------------------------------------------------------------
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    out = cv2.VideoWriter("output_hand_noisy.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, (int(width), int(height)))
    frame_init = cv2.cvtColor(frame_init_rgb, cv2.COLOR_BGR2GRAY).astype('float64')
    model = frame_init[y_start:y_start+size_y, x_start:x_start+size_x]
    sample_space = (0, 0, width, height)

    tracker = ParticleTracker(model, sample_space, std_control=10, n_particles=700, std_MSE=5, alpha=0.2)
    count = 0
    while(cap.isOpened()):
        ret, frame_rgb = cap.read()
        frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY).astype('float64')
        if not ret:
            break # unsuccessful reading 

        tracker.update(frame_gray)
        frame_draw = tracker.visualize_frame(frame_rgb)
        
        cv2.imshow('image', frame_draw)
        out.write(frame_draw)
        
        delay = int(25 - (time.time() - start_time))
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    
        count += 1
        #if count == 15 or count == 50 or count == 140:
            #cv2.imwrite(str(count)+'_hand_noisy.png', frame_draw)
    
    cap.release()
    cv2.destroyAllWindows()
    