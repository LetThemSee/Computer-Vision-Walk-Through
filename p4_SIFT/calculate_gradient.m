function [X_grad, Y_grad] = calculate_gradient(img_smooth)
X_grad = imfilter(img_smooth, [-1, 1]);
Y_grad = imfilter(img_smooth, [-1; 1]);