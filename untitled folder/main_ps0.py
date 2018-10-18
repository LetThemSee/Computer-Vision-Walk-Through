#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 08:04:38 2018

@author: jinzhao
"""

import numpy as np
import cv2

def get_center(img):
    n_row, n_col = img.shape
    c_x = n_col // 2
    c_y = n_row // 2
        
    return c_x, c_y

if __name__ == "__main__" :

    # PS0-1
    # Load an color image in grayscale
    img1 = cv2.imread('input/planet.jpg', 1)
    img2 = cv2.imread('input/motorcycle.jpg', 1)

    cv2.imshow('image1', img1)
    cv2.imshow('image2', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # PS0-2
    # BGR order if you used cv2.imread(),
    img1 = img1[:, :, [2, 1, 0]]
    img1_G = img1[:, :, 1]
    img1_R = img1[:, :, 0]
    
    cv2.imshow('img', img1)
    cv2.imshow('imgG', img1_G)
    cv2.imshow('imgR', img1_R)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # PS0-3
    c_x1, c_y1 = get_center(img1_G)
    center_patch = img1_G[c_y1-49 : c_y1+50, c_x1-49 : c_x1+50] 
    
    img2_G = img2[:, :, 1]
    c_x2, c_y2 = get_center(img2_G)
    img2_G[c_y2-49 : c_y2+50, c_x2-49 : c_x2+50] = center_patch

    
    cv2.imshow('center', center_patch)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('img2_new', img2_G)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # PS0-4
    img1_G = np.float64(img1_G)
    img1_G = (img1_G - np.mean(img1_G)) / np.std(img1_G) * 10
    img1_G = np.uint8(img1_G)
    
    cv2.imshow('img1_G', img1_G)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    img1_G_new = np.column_stack((img1_G, np.zeros((img1_G.shape[0], 2))))
    img1_G_new = img1_G_new[:, 2:]
    
    cv2.imshow('img1_G_new', np.uint8(img1_G_new))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    diff = img1_G - img1_G_new
    
    cv2.imshow('diff', diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # PS0-5
    img1 = cv2.imread('input/planet.jpg', 1)
    img1_green = img1[:, :, 1]
    n_row, n_col = img1_green.shape 
    
    noise_sigma = 0.4 # sigma = 0.3 ~ 0.4
    noise = np.random.randn(n_row, n_col) * noise_sigma
    
    imgG_noise = img1_green + noise

    img1_blue = img1[:, :, 0] # reload the image in BGR
    imgB_noise = img1_blue + noise
    
    cv2.imshow('noiseG', np.uint8(imgG_noise))
    cv2.imshow('noiseB', np.uint8(imgB_noise))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    

    
    
    
    






