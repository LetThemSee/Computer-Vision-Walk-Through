#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 12:55:36 2018

@author: jinzhao
"""
import numpy as np
import cv2
import os

L = cv2.imread(os.path.join('input', 'pair1-L.png'), 0) * (1.0 / 255.0) 
R = cv2.imread(os.path.join('input', 'pair1-R.png'), 0) * (1.0 / 255.0)
# TODO: Your code here
height, width = L.shape
D = np.zeros(L.shape)

win_size = 5
half_size = win_size // 2

disparity_size = 50
half_disp = disparity_size // 2

for row in range(half_size, height-half_size):
    for col in range(half_size, width-half_size):
        # 1.Get a template
        tpl = L[row-half_size:row+half_size+1, col-half_size:col+half_size+1].astype(np.float32)
        # 2.Localize the scan line
        region_col_min = max(col-half_disp, 0)
        region_col_max = min(col+half_disp+1, width)
        cor_region = R[row-half_size:row+half_size+1, :].astype(np.float32)
        # 3.Search for the best match along this scan line
        result = cv2.matchTemplate(cor_region, tpl, cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
        D[row, col] = min_loc[0]+ half_size - col
        
D_tmp = D[half_size:height-half_size, half_size:width-half_size]
D_norm = cv2.normalize(D_tmp, D_tmp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)




