#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 19:29:56 2018

@author: jinzhao
"""

# ps3

import cv2
import numpy as np
import os
import time
import random
from ps3_functions import *

if __name__ == "__main__":
    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ 
    start_time = time.time()
    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ 
    
    '''
    ## 1-a
    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ 
    pts_2d_norm = read_points('input/pts2d-norm-pic_a.txt', 2)
    pts_3d_norm = read_points('input/pts3d-norm.txt', 3)
    
    M = compute_M(pts_2d_norm, pts_3d_norm)
    
    test_2d = np.array([[0.1419], [-0.4518]])
    test_3d = np.array([[1.2323], [1.4421], [0.4506], [1.0]])
    proj_2d = np.dot(M, test_3d)
    proj_2d = proj_2d[0:2] / proj_2d[2] 
    
    residual = compute_residual(proj_2d, test_2d)
    '''
    
    '''
    ## 1-b
    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ 
    pts_2d_norm = read_points('input/pts2d-norm-pic_a.txt', 2)
    pts_2d = read_points('input/pts2d-pic_a.txt', 2)
    pts_3d = read_points('input/pts3d.txt', 3)
    
    n_total_pts = pts_2d.shape[1]
    
    n_pts_set = [8, 12, 16]
    M_opt_set = []
    ave_res = np.zeros((10, 3))
    for i in range(len(n_pts_set)):
        n_pts = n_pts_set[i]
        M_set = []
        for j in range(10):
            rand_set = random.sample(range(n_total_pts), n_pts+4)
            
            pts_2d_tmp = pts_2d[:, rand_set[0:n_pts]]
            pts_3d_tmp = pts_3d[:, rand_set[0:n_pts]]
            
            M = compute_M(pts_2d_tmp, pts_3d_tmp)
            M_set.append(M)
            
            pts_2d_test = pts_2d_norm[:, rand_set[-4:]]
            
            pts_3d_test = np.append(pts_3d[:, rand_set[-4:]], np.ones((1, 4)), axis=0)
            pts_2d_proj_homo = np.dot(M, pts_3d_test)
            pts_2d_proj = pts_2d_proj_homo[0:2] / pts_2d_proj_homo[2]
            
            res = compute_residual(pts_2d_proj, pts_2d_test)
            ave_res[j, i] = np.mean(res)
            
        min_idx = np.argmin(ave_res[:, i])
        M_opt_set.append(M[min_idx])
    '''  
    
    '''
    ## 1-c
    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ 
    pts_2d_norm = read_points('input/pts2d-norm-pic_a.txt', 2)
    pts_3d_norm = read_points('input/pts3d-norm.txt', 3)
    M_normA = compute_M(pts_2d_norm, pts_3d_norm)
    
    Q = M_normA[:, 0:3]
    m4 = M_normA[:, 3]
    
    C_normA = - np.dot(np.linalg.inv(Q), m4)
    '''
    
    '''
    ## 2-a
    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ 
    pts_a = read_points('input/pts2d-pic_a.txt', 2)
    pts_b = read_points('input/pts2d-pic_b.txt', 2)
    
    F = compute_F(pts_a, pts_b, False)
        
    ## 2-a
    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ 
    
    F_fixed = compute_F(pts_a, pts_b, True)
    
    ## 2-c
    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ 
    pic_a = cv2.imread(os.path.join('input', 'pic_a.jpg'), 1) 
    pic_b = cv2.imread(os.path.join('input', 'pic_b.jpg'), 1)  
    #img_show([pic_a, pic_b])
    
    pic_b_epi = draw_epipolar_lines(pts_a, F_fixed, pic_b)
    pic_a_epi = draw_epipolar_lines(pts_b, F_fixed.T, pic_a) # Transpose of F
    img_show([pic_b_epi, pic_a_epi])
    
    #cv2.imwrite('output/ps3-2-c-1.png', pic_a_epi)
    #cv2.imwrite('output/ps3-2-c-2.png', pic_b_epi)
    '''
    
    ## Extra
    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ 
    pts_a = read_points('input/pts2d-pic_a.txt', 2)
    pts_b = read_points('input/pts2d-pic_b.txt', 2)

    T_a, pts_a_norm = compute_norm_transform(pts_a)
    T_b, pts_b_norm = compute_norm_transform(pts_b)    
    
    F = compute_F(pts_a_norm, pts_b_norm, False)
    F_fixed = compute_F(pts_a_norm, pts_b_norm, True)
    
    F_new = np.dot(T_b.T, np.dot(F_fixed, T_a))

    pic_a = cv2.imread(os.path.join('input', 'pic_a.jpg'), 1) 
    pic_b = cv2.imread(os.path.join('input', 'pic_b.jpg'), 1)  
    pic_b_epi = draw_epipolar_lines(pts_a, F_new, pic_b)
    pic_a_epi = draw_epipolar_lines(pts_b, F_new.T, pic_a) # Transpose of F
    img_show([pic_b_epi, pic_a_epi])
    
   
    
    
    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ 
    elapsed_time = time.time() - start_time
    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ 