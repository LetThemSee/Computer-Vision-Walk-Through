#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 10:20:23 2018

@author: jinzhao
"""
import numpy as np
import cv2

def img_show(images):
    if len(images) > 5:
        raise ValueError('A very specific bad thing happened.')
    for i in range(len(images)):
        img = np.uint8(images[i])
        cv2.imshow('img'+str(i), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def Canny_edge(image, sigma=0.33):
    med = np.median(image)
    low = int(max(0, (1.0 - sigma) * med))
    high = int(min(255, (1.0 + sigma) * med))
    img_edge = cv2.Canny(image, low, high)
    
    return img_edge
    
# ====== ====== ====== ====== ====== ====== ====== ====== ====== ====== ====== 
# Lines
# ====== ====== ====== ====== ====== ====== ====== ====== ====== ====== ====== 
def hough_lines_acc(img_edges, step_size = 1):
    step_size = 1
    height, width = img_edges.shape
    
    theta_range = range(-90, 90)
    n_theta = len(theta_range)
    
    d_max = int(np.sqrt(width ** 2 + height ** 2))
    diag = int(d_max / step_size) 
    d_range = range(-diag, diag+1, step_size) # Range of 'd': -diag -> 0 -> +diag
    n_d = len(d_range) 
    
    H = np.zeros((n_d, n_theta))
    
    for x in range(width):
        for y in range(height):
            # For each edge point
            if img_edges[y, x] == 255:
                for idx_th in range(n_theta):
                    th_rad = theta_range[idx_th] / 180 * np.pi
                    d = (x+1) * np.cos(th_rad) + (y+1) * np.sin(th_rad) # could be negative
                    
                    idx_d = int(d / step_size) + diag
                    
                    H[idx_d, idx_th] += 1
    
    return H, d_range, theta_range

    
def hough_peaks(H, n_peaks_atmost, threshold=None, hood_size=None):
    
    if threshold is None:
        threshold = 0.5 * np.max(H)
        
    n_row, n_col = H.shape
    if hood_size is None:
        half_hood_row = int(n_row / 100) # hood_row = int(n_row/100) * 2 + 1 !Default
        half_hood_col = int(n_col / 100) # hood_col = int(n_col/100) * 2 + 1 !Default
    else:
        half_hood_row = hood_size[0]
        half_hood_col = hood_size[1]
    
    H_output = H * (H > threshold)   
    row_idx, col_idx = np.nonzero(H_output)
    n_peaks = len(row_idx)
    
    values = np.zeros(n_peaks)
    for i in range(n_peaks):
        values[i] = H[row_idx[i], col_idx[i]]
    #values_sort = np.sort(values)[::-1] # Descending order
    index_sort = np.argsort(values)[::-1]
    
    # Non-maximum suppression
    peaks_output = np.empty((0, 2), dtype='int64')
    
    for i in range(n_peaks):
        idx_peak = index_sort[i]
        row = row_idx[idx_peak]
        col = col_idx[idx_peak]
        hood = H[max(row-half_hood_row, 0) : min(row+half_hood_row+1, n_row), \
                 max(col-half_hood_col, 0) : min(col+half_hood_col+1, n_col)] # boundary test
        if H_output[row, col] == np.max(hood):  
            peaks_output = np.append(peaks_output,  np.array([[row, col]], dtype='int64'), axis = 0)
            
        if peaks_output.shape[0] == n_peaks_atmost:
            break # Must not exceed the (required) maximum number of peaks
            
    if len(peaks_output) == 0:
        print("No peaks found")
        return
    
    return peaks_output

def hough_lines_draw(img_rgb, peaks, d_range, theta_range):

    line = np.max(d_range) * 10 # Infinite line 
    n_peaks = peaks.shape[0]
    for i in range(n_peaks):
        rho = d_range[int(peaks[i, 0])]
        theta = theta_range[int(peaks[i, 1])] * np.pi / 180
        a = np.cos(theta)
        b = np.sin(theta)
        pt = rho * np.array([a, b])
        pt_start = tuple((pt + line * np.array([-b, a])).astype(int))
        pt_end = tuple((pt - line * np.array([-b, a])).astype(int))
         
        cv2.line(img_rgb, pt_start, pt_end, (0, 255, 0), 2)
        
    return img_rgb

def filter_lines(peaks, theta_range, rho_range, theta_threshold, rho_threshold):
    n_peaks = peaks.shape[0]
    
    del_list = []
    for i in range(n_peaks):
        delta_rho = np.abs(np.array([abs(rho_range[peaks[j, 0]] - rho_range[peaks[i, 0]])
                              for j in range(len(peaks))]))

        delta_theta = np.array([abs(theta_range[peaks[j, 1]] - theta_range[peaks[i, 1]])
                                for j in range(len(peaks))])

        if not ((delta_theta < theta_threshold) & (delta_rho > 1) &
                (delta_rho < rho_threshold)).any():
            del_list.append(i)

    peaks_filtered = np.delete(peaks, del_list, 0)
    
    return peaks_filtered

def find_lines(img_edges, n_peaks_atmost, img_rgb, step_size=1, threshold_factor=None, hood_size=None):
    H, d_range, theta_range = hough_lines_acc(img_edges, step_size)
    peaks_output = hough_peaks(H, n_peaks_atmost, threshold_factor, hood_size)
    img_rgb = hough_lines_draw(img_rgb, peaks_output, d_range, theta_range)
    
    return H, peaks_output, img_rgb


# ====== ====== ====== ====== ====== ====== ====== ====== ====== ====== ====== 
# Circles
# ====== ====== ====== ====== ====== ====== ====== ====== ====== ====== ====== 
   
def hough_circles_acc(img_edges, radius): # Only handle one given radius
    theta_range = range(-180, 180)
    
    height, width = img_edges.shape   
    H = np.zeros(img_edges.shape)
   
    for x in range(width):
        for y in range(height):
            # For every edge pixel (x,y) :
            if img_edges[y, x] == 255:
                for degree in theta_range:
                    rad = degree / 180 * np.pi
                    a = int(round(x - radius * np.cos(rad)))
                    b = int(round(y + radius * np.sin(rad)))
                    H[b, a] += 1
    return H

def hough_circle_draw(img_rgb, centers, radius):
    for i in range(centers.shape[0]):
        x = centers[i, :].astype(int)[1]
        y = centers[i, :].astype(int)[0]
        cv2.circle(img_rgb, tuple((x, y)), radius, (0, 255, 0))
    return img_rgb

def find_circles(img_edges, radius_range, img_rgb, threshold = 125, n_peaks_atmost=20, hood_size=(10,10)):
    # 1. Hough accumulator
    theta_range = range(-180, 180)
    
    height, width = img_edges.shape 
    depth = len(radius_range)
    H = np.zeros((height, width, depth)) 
   
    for x in range(width):
        for y in range(height):
            # For every edge pixel (x,y) :
            if img_edges[y, x] == 255:
                # For each possible radius value r
                for c in range(len(radius_range)):
                    radius = radius_range[c]
                    for degree in theta_range:
                        rad = degree / 180 * np.pi
                        a = int(round(x - radius * np.cos(rad)))
                        b = int(round(y + radius * np.sin(rad)))
                        if a > 0 and a < width:
                            if b > 0 and b < height:
                                H[b, a, c] += 1
    # 2. Hough peaks
    half_hood_row = hood_size[0] # hood_row = int(n_row/100) * 2 + 1 !Default
    half_hood_col = hood_size[1]# hood_col = int(n_col/100) * 2 + 1 !Default
    
    H_output = H * (H > threshold)   
    row_idx_set, col_idx_set, dep_idx_set = np.nonzero(H_output)
    n_peaks = len(row_idx_set)
    
    values = np.zeros(n_peaks)
    for i in range(n_peaks):
        values[i] = H[row_idx_set[i], col_idx_set[i], dep_idx_set[i]]
    index_sort = np.argsort(values)[::-1] # Descending order
    
    
    
    # Non-maximum suppression
    peaks_output = np.empty((0, 2), dtype='int64')
    radii_output = np.empty((0, 1), dtype='int64')
    
    for i in range(n_peaks):
        idx_peak = index_sort[i]
        row = row_idx_set[idx_peak]
        col = col_idx_set[idx_peak]
        dep = dep_idx_set[idx_peak]
        # Caution: Avoid exceeding boundary
        hood = H_output[max(row-half_hood_row, 0) : min(row+half_hood_row+1, height), \
                 max(col-half_hood_col, 0) : min(col+half_hood_col+1, width), :]  # Only consider the case where one center has one radius.
        if H[row, col, dep] == np.max(hood):  
            radius = radius_range[dep]

            peaks_output = np.append(peaks_output, np.array([[row, col]], dtype='int64'), axis = 0)
            radii_output = np.append(radii_output, np.array([[radius]], dtype='int64'), axis = 0)
        
        if peaks_output.shape[0] == n_peaks_atmost:
            break
    
    # 3. Draw circles
    n_centers = len(peaks_output)
    for i in range(n_centers):
        x = peaks_output[i, 1]
        y = peaks_output[i, 0]
        radius = radii_output[i]
        cv2.circle(img_rgb, tuple((x, y)), radius, (0, 255, 0), 2)
    
    return peaks_output, radii_output, img_rgb

