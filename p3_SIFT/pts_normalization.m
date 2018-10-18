function [pts_n, norm_mat] = pts_normalization(pts_homo)
% Input: pts_homo 3 x n_pts

pts_mean = mean(pts_homo, 2);
x_mean = pts_mean(1);
y_mean = pts_mean(2);

% Compute scale factor s:
% Method 1:
pts_tmp = pts_homo - pts_mean;

pts_std = std(pts_tmp);
x_std = pts_std(1);
y_std = pts_std(2);

s_x = 1/x_std;
s_y = 1/y_std;

% Method 2:
s_x = max(abs(pts_homo(1, :)));
s_y = max(abs(pts_homo(2, :)));

scale = [s_x, 0, 0
         0, s_y, 0
         0, 0, 1];

offset = [1, 0, -x_mean
          0, 1, -y_mean
          0, 0, 1];

norm_mat = scale * offset;

pts_n = norm_mat * pts_homo;

