import numpy as np
import cv2

def disparity_ssd(L, R, win_size=7, disparity_size=150):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
    
    Params:
    L: Grayscale left image
    R: Grayscale right image, same size as L

    Returns: Disparity map, same size as L, R
    """
    # TODO: Your code here
    height, width = L.shape
    D = np.zeros(L.shape)
    
    half_size = win_size // 2
    
    half_disp = disparity_size // 2
    
    for row in range(half_size, height-half_size):
        for col in range(half_size, width-half_size):
            '''
            ### My way of implementing SSD
            # 1.Get a template
            tpl = L[row-half_size : row+half_size+1, col-half_size : col+half_size+1]
            
            # 2.Search for the best match along the same scan line
            SSD = np.ones((1, width)) * np.inf

            for x in range(half_size, width-half_size):
                cur_tpl = R[row-half_size : row+half_size+1, x-half_size : x+half_size+1]
                SSD[0, x] = np.sum((tpl - cur_tpl)**2)
                
            # 3.Calculate the offset   
            min_idx = np.argmin(SSD)
            if abs(min_idx-col) < 0.02 * width:
                D[row, col] = min_idx - col
            '''
            ### OpenCV is way faster, which I am not happy about.
            # 1.Get a template
            tpl = L[row-half_size:row+half_size+1, col-half_size:col+half_size+1].astype(np.float32)
            # 2.Localize the scan line
            region_col_min = max(col-half_disp, 0)
            region_col_max = min(col+half_disp+1, width)
            cor_region = R[row-half_size:row+half_size+1, region_col_min:region_col_max].astype(np.float32)
            #cor_region = R[row-half_size:row+half_size+1, :].astype(np.float32)region_col_min
            # 3.Search for the best match along this scan line
            result = cv2.matchTemplate(cor_region, tpl, cv2.TM_SQDIFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
            D[row, col] = min_loc[0] + region_col_min + half_size - col
            
    '''
    D_tmp = D[half_size:height-half_size, half_size:width-half_size]
    D_scale = (D_tmp - np.min(D_tmp)) / (np.max(D_tmp) - np.min(D_tmp)) * 255
    '''
    D_scale = cv2.normalize(D, D, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return D, D_scale
    