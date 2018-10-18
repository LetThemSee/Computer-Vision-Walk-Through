function [keypoints, angles] = Harris_Corner(img_smooth, alpha, win_size, threshold_factor, hood_size)
[X_grad, Y_grad] = calculate_gradient(img_smooth);
[R, R_norm] = ps4_1_b_Harris_value(X_grad, Y_grad, alpha, win_size);

keypoints = ps4_1_c_Harris_Corner(R, threshold_factor, hood_size);
angles = ps4_2_a_Angles(keypoints, X_grad, Y_grad);
