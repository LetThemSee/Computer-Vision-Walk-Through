function [X_grad_img, Y_grad_img] = ps4_1_a(img_smooth)

X_grad_img = map_gradient(imfilter(img_smooth, [-1, 1]));
Y_grad_img = map_gradient(imfilter(img_smooth, [-1; 1]));


