#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 10:30:10 2018

@author: jinzhao
"""
import numpy as np
import cv2

def read_points(file_name, dim):
    pts = np.empty((dim, 0))
    with open(file_name) as file:
        for line in file:
            line = line.strip().split()
            point = np.float64(line).reshape(dim, 1)
            pts = np.append(pts, point, axis=1)
            
    return pts

def compute_M(pts_2d, pts_3d):
    n_pts = pts_2d.shape[1]
    # Construct matrix A
    A = np.zeros((2*n_pts, 12))
    for i in range(n_pts):
        tmp = np.append(pts_3d[:, i], 1).reshape(1, 4) 
        A[2*i, 0:4] = tmp 
        A[2*i, 8:12] = -pts_2d[0, i] * tmp

        A[2*i+1, 4:8] = tmp
        A[2*i+1, 8:12] = -pts_2d[1, i] * tmp
    
    # Use SVD to Compute m
    U, D, VT = np.linalg.svd(A, full_matrices=True)
    V = VT.T
    m = V[:, -1]
    
    # Construct matrix M
    M = np.zeros((3, 4))
    for i in range(3):
        M[i, :] = m[4*i : 4*i+4]
    
    return M

def compute_residual(pts_2d_proj, pts_2d_test):
    n_pts = pts_2d_proj.shape[1]
    residual = [0] * n_pts
    
    for i in range(n_pts):
        res = np.linalg.norm(pts_2d_proj[:, i] - pts_2d_test[:, i])
        residual[i] = res
    
    return residual

def compute_F(pts_a, pts_b, fixed=False):
    n_pts = pts_a.shape[1]
    
    # Construct matrix A.(Same notation but a different A from 'compute_M')
    A = np.zeros((n_pts, 9))
    for i in range(n_pts):
        tmp = np.append(pts_a[:, i], 1).reshape(1, 3) 
        A[i, 0:3] = pts_b[0, i] * tmp
        A[i, 3:6] = pts_b[1, i] * tmp 
        A[i, 6:9] = tmp 
    
    # Use SVD to Compute m
    U, D, VT = np.linalg.svd(A, full_matrices=True)
    V = VT.T
    f = V[:, -1]
    
    # Construct matrix M
    F = np.zeros((3, 3))
    for i in range(3):
        F[i, :] = f[3*i : 3*i+3]
    
    if fixed is True:
        U, D, VT = np.linalg.svd(F, full_matrices=True)
        D_ = np.diag(D)
        D_[-1, -1] = 0
        F = np.dot(np.dot(U, D_), VT)
    
    return F

def img_show(images):
    if len(images) > 5:
        raise ValueError('A very specific bad thing happened.')
    for i in range(len(images)):
        img = np.uint8(images[i])
        cv2.imshow('img'+str(i), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_epipolar_lines(pts_a, F, pic_b):
    height, width, depth = pic_b.shape
    
    point_UL = [0, 0, 1]
    point_BL = [0, height-1, 1]
    line_L = np.cross(point_UL, point_BL)
    
    point_UR = [width-1, 0, 1]
    point_BR = [width-1, height-1, 1]
    line_R = np.cross(point_UR, point_BR)
    
    n_pts = pts_a.shape[1]
    pts_a_homo = np.append(pts_a, np.ones((1, n_pts)), axis=0)
    lines_b = np.dot(F, pts_a_homo)
    
    n_lines = n_pts
    for i in range(n_lines):
        pt_left_homo = np.cross(line_L, lines_b[:, i])
        pt_right_homo = np.cross(line_R, lines_b[:, i])
        
        pt_left = (pt_left_homo[0:2] / pt_left_homo[2]).astype(int)
        pt_right = (pt_right_homo[0:2] / pt_right_homo[2]).astype(int)
        
        cv2.line(pic_b, tuple(pt_left), tuple(pt_right), (0, 255, 0), 2)
    
    return pic_b

  
def compute_norm_transform(pts):
    c_u = np.mean(pts[0, :])
    c_v = np.mean(pts[1, :])
    
    offset_matrix = np.eye(3)
    offset_matrix[0, 2] = -c_u
    offset_matrix[1, 2] = -c_v
    
    pts_tmp = pts - np.array([[c_u], [c_v]])
    
    x_max = np.max(np.abs(pts_tmp[:, 0]))
    y_max = np.max(np.abs(pts_tmp[:, 1]))
    x_std = np.std(pts_tmp[:, 0])
    y_std = np.std(pts_tmp[:, 1])

    
    scale_matrix = np.diag([1/x_max, 1/y_max, 1])
    
    T = np.dot(scale_matrix, offset_matrix)
    
    n_pts = pts.shape[1]
    pts_homo = np.append(pts, np.ones((1, n_pts)), axis=0)
    pts_n_homo = np.dot(T, pts_homo)
    pts_n_homo = pts_n_homo[0:2, :]
    
    return T, pts_n_homo