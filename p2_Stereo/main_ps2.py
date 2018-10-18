# ps2
import os
import time
import numpy as np
import cv2
from disparity_ssd import disparity_ssd
from disparity_ncorr import disparity_ncorr

def img_show(images):
    if len(images) > 5:
        raise ValueError('A very specific bad thing happened.')
    for i in range(len(images)):
        img = np.uint8(images[i])
        cv2.imshow('img'+str(i), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ 
    start_time = time.time()
    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
    
    '''
    ## 1-a
    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ 
    # Read images
    L = cv2.imread(os.path.join('input', 'pair0-L.png'), 0) * (1.0 / 255.0)  # grayscale, [0, 1]
    R = cv2.imread(os.path.join('input', 'pair0-R.png'), 0) * (1.0 / 255.0)
    
    # Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
    D_L, D_L_scale = disparity_ssd(L, R)
    D_R, D_R_scale = disparity_ssd(R, L)
    
    img_show(np.uint8([D_L_scale, D_R_scale]))
    #cv2.imwrite('output/ps2-1-a-1.png', D_L_scale)
    #cv2.imwrite('output/ps2-1-a-2.png', D_R_scale)
    
    
    ## 2-a
    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ 
    L = cv2.imread(os.path.join('input', 'pair1-L.png'), 0) * (1.0 / 255.0) 
    R = cv2.imread(os.path.join('input', 'pair1-R.png'), 0) * (1.0 / 255.0)
    #img_show(np.uint8([L*255, R*255]))
    
    # Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
    
    D_L, D_L_scale = disparity_ssd(L, R)
    D_R, D_R_scale = disparity_ssd(R, L)
    
    img_show([D_L_scale, D_R_scale])
    #cv2.imwrite('output/ps2-2-a-1.png', D_L_scale)
    #cv2.imwrite('output/ps2-2-a-2.png', D_R_scale)
    
    
    ## 3-a
    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
    L = cv2.imread(os.path.join('input', 'pair1-L.png'), 0) * (1.0 / 255.0)  # grayscale, [0, 1]
    R = cv2.imread(os.path.join('input', 'pair1-R.png'), 0) * (1.0 / 255.0)
    
    gauss_noise = np.random.normal(0, 0.05, L.shape)
    L_noise = L + gauss_noise
    # L_noise_img = L_noise * 255
    #img_show([L_noise_img])
    #cv2.imwrite('output/ps2-3-noise.png', L_noise_img)
    
    D_Ln, D_Ln_scale = disparity_ssd(L_noise, R)
    D_Rn, D_Rn_scale = disparity_ssd(R, L_noise)
    img_show([D_Ln_scale, D_Rn_scale])
    #cv2.imwrite('output/ps2-3-a-1.png', D_Ln_scale)
    #cv2.imwrite('output/ps2-3-a-2.png', D_Rn_scale)
    
    
    ## 3-b
    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
    L_mul = L * 1.1
    # L_mul_img = L_noise * 255
    #img_show([L_mul_img])
    #cv2.imwrite('output/ps2-3-mul.png', L_mul_img)
    
    
    D_Lm, D_Lm_scale = disparity_ssd(L_mul, R)
    D_Rm, D_Rm_scale = disparity_ssd(R, L_mul)
    #img_show([D_Lm_scale, D_Rm_scale])
    #cv2.imwrite('output/ps2-3-b-1.png', D_Lm_scale)
    #cv2.imwrite('output/ps2-3-b-2.png', D_Rm_scale)
    
    
    
    ## 4-a
    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
    L = cv2.imread(os.path.join('input', 'pair1-L.png'), 0) * (1.0 / 255.0)  # grayscale, [0, 1]
    R = cv2.imread(os.path.join('input', 'pair1-R.png'), 0) * (1.0 / 255.0)
    
    D_L, D_L_scale = disparity_ncorr(L, R)
    D_R, D_R_scale = disparity_ncorr(R, L)
    
    img_show([D_L_scale, D_R_scale])
    cv2.imwrite('output/ps2-4-a-1.png', D_L_scale)
    cv2.imwrite('output/ps2-4-a-2.png', D_R_scale)
    
    
    ## 4-b
    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
    D_Ln, D_Ln_scale = disparity_ncorr(L_noise, R)
    D_Rn, D_Rn_scale = disparity_ncorr(R, L_noise)
    img_show([D_Ln_scale, D_Rn_scale])
    cv2.imwrite('output/ps2-4-b-1.png', D_Ln_scale)
    cv2.imwrite('output/ps2-4-b-2.png', D_Rn_scale)
    
    D_Lm, D_Lm_scale = disparity_ncorr(L_mul, R)
    D_Rm, D_Rm_scale = disparity_ncorr(R, L_mul)
    img_show([D_Lm_scale, D_Rm_scale])
    cv2.imwrite('output/ps2-4-b-3.png', D_Lm_scale)
    cv2.imwrite('output/ps2-4-b-4.png', D_Rm_scale)
    '''
    
    ## 5
    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
    L = cv2.imread(os.path.join('input', 'pair2-L.png'), 0) * (1.0 / 255.0)  # grayscale, [0, 1]
    R = cv2.imread(os.path.join('input', 'pair2-R.png'), 0) * (1.0 / 255.0)
    
    D_L, D_L_scale = disparity_ncorr(L, R)
    #D_R, D_R_scale = disparity_ssd(R, L)
    
    img_show([D_R_scale])
    #cv2.imwrite('output/ps2-5-a-1.png', D_L_scale)
    #cv2.imwrite('output/ps2-5-a-2.png', D_R_scale)
    
    '''
    
    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ 
    elapsed_time = time.time() - start_time
    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ 
    