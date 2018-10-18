close all; clear; clc;

%% ps4-1-a Gradient
% img = double(imread(fullfile('input', 'transA.jpg')));
% img_smooth = imgaussfilt(img, 1);

% [X_grad_uint8, Y_grad_uint8] = ps4_1_a(img_smooth);
% 
% grad_pair = horzcat(X_grad_uint8, Y_grad_uint8);
% figure, imshow(uint8(grad_pair)); title('gradient pair');
% %imwrite(grad_pair, fullfile('output', 'ps4-1-a-2.png'));
%% ps4-1-b Harris Value
% img = double(imread(fullfile('input', 'simB.jpg')));
% img_smooth = imgaussfilt(img, 0.5);
% [X_grad, Y_grad] = calculate_gradient(img_smooth);
% alpha = 0.06;
% win_size = 5;
% [R, R_norm] = ps4_1_b_Harris_value(X_grad, Y_grad, alpha, win_size);
% 
% %figure, imshow(uint8(R_norm))
% %imwrite(uint8(R_norm), fullfile('output', 'ps4-1-b-1.png'));
% %% ps4-1-c Harris Corner
% threshold_factor = 0.01;
% hood_size = 11;
% keypoints = ps4_1_c_Harris_Corner(R, threshold_factor, hood_size);
% 
% %plot_corners(img, keypoints);
% %saveas(gcf, fullfile('output', 'ps4-1-c-1.png'));
% %% ps4-2-a Angles
% angles = ps4_2_a_Angles(keypoints, X_grad, Y_grad);

% plot_gradient(img, keypoints, angles)
% %saveas(gcf, fullfile('output', 'ps4-2-a-4.png'));
%% ps4-2-b SIFT descriptor
% 1.Find descriptor
imgA = double(imread(fullfile('input', 'simA.jpg')));
imgB = double(imread(fullfile('input', 'simB.jpg')));
pair = horzcat(imgA, imgB);
img_smoothA = imgaussfilt(imgA, 0.5);
img_smoothB = imgaussfilt(imgB, 0.5);

alpha = 0.06;
win_size = 5;
threshold_factor = 0.01;
hood_size = 11;

[keypointsA, anglesA] = Harris_Corner(img_smoothA, alpha, win_size, ...
                                                threshold_factor, hood_size);

[F_outA, D_outA] = ps4_2_b_SIFT_descriptor(imgA, keypointsA, anglesA);

[keypointsB, anglesB] = Harris_Corner(img_smoothB, alpha, win_size, ...
                                                threshold_factor, hood_size);

[F_outB, D_outB] = ps4_2_b_SIFT_descriptor(imgB, keypointsB, anglesB);

% 2.Find match 
[matches, scores] = vl_ubcmatch(D_outA, D_outB);

% 3.Plot
%plot_matches(pair, F_outA, F_outB, matches, imgA)
%saveas(gcf, fullfile('output', 'ps4-2-b-2.png'));
%% ps4-3-a RANSAC trans
% toler = 20;
% N = 100;
% opt_conse = compute_trans_consensus(N, toler, F_outA, F_outB, matches);
% 
% plot_matches(pair, F_outA, F_outB, opt_conse, imgA)
%saveas(gcf, fullfile('output', 'ps4-3-a-1.png'));
%% ps4-3-b RANSAC similarity
% N = 100;
% toler = 10;
% 
% [opt_conse, opt_sim] = compute_sim_consensus(N, toler, F_outA, F_outB, matches);
% plot_matches(pair, F_outA, F_outB, opt_conse, imgA);
% saveas(gcf, fullfile('output', 'ps4-3-b-1.png'));
%% ps4-3-c RANSAC affine
N = 100;
toler = 25;

[opt_conse, opt_affine] = compute_affine_consensus(N, toler, F_outA, F_outB, matches);
% 
plot_matches(pair, F_outA, F_outB, opt_conse, imgA);
% 
% saveas(gcf, fullfile('output', 'ps4-3-c-1.png'));
%% ps4-3-d Warp image (sim)
% % Inverse warping
% warpedB = inverse_warping(imgA, imgB, opt_sim);
% figure, imshow(uint8(warpedB));
% %saveas(gcf, fullfile('output', 'ps4-3-d-1.png'));
% % blend them
% [height, width] = size(imgA);
% img_rgb = zeros(height, width, 3);
% 
% img_rgb(:,:,1) = imgA;
% img_rgb(:,:,2) = warpedB;
% figure, imshow(uint8(img_rgb));
% %saveas(gcf, fullfile('output', 'ps4-3-d-2.png'));

%% ps4-3-d Warp image (affine)
warpedB = inverse_warping(imgA, imgB, opt_affine);
figure, imshow(uint8(warpedB));
saveas(gcf, fullfile('output', 'ps4-3-e-1.png'));
% blend them
[height, width] = size(imgA);
img_rgb = zeros(height, width, 3);

img_rgb(:,:,1) = imgA;
img_rgb(:,:,2) = warpedB;
figure, imshow(uint8(img_rgb));
saveas(gcf, fullfile('output', 'ps4-3-e-2.png'));