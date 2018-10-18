#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 10:03:24 2018

@author: jinzhao
"""

import numpy as np
import cv2
from ps1_functions import *
        
if __name__ == "__main__":
    '''
    #p4-1
    #------ ------ ------ ------ ------ ------ ------ ------ ------ ------ 
    img = cv2.imread('input/ps1-input0.png', 0)
    img_edges = cv2.Canny(img, 100, 200)
    
    #img_show([img, img_edges])
    #cv2.imwrite('output/ps1-1-a-1.png', img_edges)
    
    #p4-2
    #------ ------ ------ ------ ------ ------ ------ ------ ------ ------ 
    H, d_range, theta_range = hough_lines_acc(img_edges, step_size = 1)
    #img_show([H])
    #cv2.imwrite('output/ps1-2-a-1.png', np.uint8(H))
    
    n_peaks_atmost = 10
    peaks = hough_peaks(H, n_peaks_atmost)
    
    img_rgb = cv2.imread('input/ps1-input0.png', 1)
    img_rgb = hough_lines_draw(img_rgb, peaks, d_range, theta_range)
    #img_show([img_rgb])
    #cv2.imwrite('output/ps1-2-c-1.png', img_rgb)
    
    #p4-3
    #------ ------ ------ ------ ------ ------ ------ ------ ------ ------ 
    img_noise = cv2.imread('input/ps1-input0-noise.png', 0)
    img_smooth = cv2.GaussianBlur(img_noise, (7, 7), 0) # 0 means sigma is calculated based on kernel size
    #cv2.imwrite('output/ ps1-3-a-1.png', img_smooth)

    edge_noise = cv2.Canny(img_noise, 150, 220)
    edge_smooth = cv2.Canny(img_smooth, 150, 220)
    #img_show([edge_noise, edge_smooth])
    #cv2.imwrite('output/ps1-3-b-1.png', edge_noise)
    #cv2.imwrite('output/ps1-3-b-2.png', edge_smooth)

    H, d_range, theta_range = hough_lines_acc(edge_smooth, step_size = 1)   
    n_peaks_atmost = 10
    peaks = hough_peaks(H, n_peaks_atmost)
    
    img_rgb = cv2.imread('input/ps1-input0-noise.png', 1)
    img_rgb = hough_lines_draw(img_rgb, peaks, d_range, theta_range)
    img_show([img_rgb])
    #cv2.imwrite('output/ps1-3-c-2.png', edge_smooth)
   
    #p4-4
    #------ ------ ------ ------ ------ ------ ------ ------ ------ ------ 
    img = cv2.imread('input/ps1-input1.png', 0)
    img_smooth = cv2.GaussianBlur(img, (7, 7), 0)
    #cv2.imwrite('output/ps1-4-a-1.png', img_smooth)
    img_edges = cv2.Canny(img_smooth, 100, 200)
    #cv2.imwrite('output/ps1-4-b-1.png', img_smooth)
    
    H, d_range, theta_range = hough_lines_acc(img_edges, step_size = 1)   
    n_peaks_atmost = 10
    peaks = hough_peaks(H, n_peaks_atmost)
    
    img_rgb = cv2.imread('input/ps1-input1.png', 1)
    img_rgb = hough_lines_draw(img_rgb, peaks, d_range, theta_range)
    #img_show([img_rgb])
    #cv2.imwrite('output/ps1-4-c-2.png', img_rgb)
    
    
    #p4-5
    #------ ------ ------ ------ ------ ------ ------ ------ ------ ------ 
    img = cv2.imread('input/ps1-input1.png', 0)
    img_smooth = cv2.GaussianBlur(img, (7, 7), 0)
    #img_show([img, img_smooth])
    #cv2.imwrite('output/ps1-5-a-1.png', img_smooth)
    
    img_edges = cv2.Canny(img_smooth, 100, 200)
    #img_show([img_edges])
    #cv2.imwrite('output/ps1-5-a-2.png', img_edge)
    
    
    # Only one radius given.
    # ------ 
    radius = 20 
    H = hough_circles_acc(img_edges, radius)
    #img_show([H])
    #cv2.imwrite('output/ps1-5-a-3-H.png', H)
    
    img_rgb = cv2.imread('input/ps1-input1.png', 1)
    n_peaks_atmost = 10
    centers = hough_peaks(H, n_peaks_atmost)
    img_rgb = hough_circle_draw(img_rgb, centers, radius)
    img_show([H, img_rgb])
    #cv2.imwrite('output/ps1-5-a-3.png', img_rgb)
    
    # Multiple radii
    # ------
    radius_range = range(20, 50+1)
    img_rgb = cv2.imread('input/ps1-input1.png', 1)
    peaks_output, radii_output, img_rgb = find_circles(img_edges, radius_range, img_rgb)
    #img_show([img_rgb])
    #cv2.imwrite('output/ps1-5-b-1.png', img_rgb)
    '''
    
    #p4-6
    #------ ------ ------ ------ ------ ------ ------ ------ ------ ------ 
    img = cv2.imread('input/ps1-input2.png', 0)
    img_smooth = cv2.GaussianBlur(img, (3, 3), 0)
    img_edges = cv2.Canny(img_smooth, 100, 210)
    img_show([img_edges])
    #cv2.imwrite('output/ps1-5-a-2.png', img_edge)
    
    #H, peaks_output, img_rgb = find_lines(img_edges, n_peaks_atmost, img_rgb)
    step_size = 1
    H, d_range, theta_range = hough_lines_acc(img_edges, step_size)
    n_peaks_atmost = 8
    peaks_output = hough_peaks(H, n_peaks_atmost)
    
    # Filter lines
    # ---
    n_peaks = peaks_output.shape[0]
    for i in range(n_peaks):
        
    
    
    peaks = peaks_output[4, :].reshape((1, 2))
    
    img_rgb = cv2.imread('input/ps1-input2.png', 1)
    img_rgb = hough_lines_draw(img_rgb, peaks, d_range, theta_range)
    img_show([img_rgb])
    #cv2.imwrite('output/ps1-5-b-1.png', img_rgb)
    
    
    
    


  
            


    

    



    
    
    
    
    

                    
                    
                
    
    