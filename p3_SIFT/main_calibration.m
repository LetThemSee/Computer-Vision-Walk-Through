%% PS3 part1
close all; clear; clc;

%% 1-a
% Read points
tmp_2d = textread(fullfile('input', 'pts2d-norm-pic_a.txt'), '%f');
tmp_3d = textread(fullfile('input', 'pts3d-norm.txt'), '%f');

n_pts = size(tmp_2d, 1) / 2;
pts_2d_n = zeros(2, n_pts);
pts_3d_n = zeros(3, n_pts);

for i = 1 : n_pts
    pts_2d_n(:, i) = [tmp_2d(2*i-1); tmp_2d(2*i)];
    pts_3d_n(:, i) = [tmp_3d(3*i-2); tmp_3d(3*i-1); tmp_3d(3*i)];
end

% Compute matrix M
M_norm = DLC_homo(pts_2d_n, pts_3d_n);

% Comparision
residual = compute_residual(pts_3d_n, pts_2d_n, M_norm);
aver_res = mean(residual);
%% 1-b 
% Read data
tmp_2d = textread(fullfile('input', 'pts2d-pic_b.txt'), '%f');
tmp_3d = textread(fullfile('input', 'pts3d.txt'), '%f');

n_pts = length(tmp_2d) / 2;
pts_2d = zeros(2, n_pts);
pts_3d = zeros(3, n_pts);

for i = 1 : n_pts
    pts_2d(:, i) = [tmp_2d(2*i-1); tmp_2d(2*i)];
    pts_3d(:, i) = [tmp_3d(3*i-2); tmp_3d(3*i-1); tmp_3d(3*i)];
end

residual_set = zeros(10, 3);
M_set = cell(10, 3);
k_set = [8, 12, 16];

for k_id = 1 : length(k_set)
    k = k_set(k_id);
    for iter = 1 : 10
        rand_idx = randperm(n_pts);
        pts_2d_picked = pts_2d(:, rand_idx(1:k));
        pts_3d_picked = pts_3d(:, rand_idx(1:k));
        M = DLC_homo(pts_2d_picked, pts_3d_picked);
        M_set{iter, k_id} = M;

        pts_2d_test = pts_2d(:, rand_idx(k+1:k+4));
        pts_3d_test = pts_3d(:, rand_idx(k+1:k+4));
        
        residual = compute_residual(pts_3d_test, pts_2d_test, M);
        residual_set(iter, k_id) = mean(residual);
    end

end

min_res = min(min(residual_set));
[row, col] = find(residual_set == min_res);
M_opt = M_set{row, col};

%% Test on M_opt
res_test = compute_residual(pts_3d_n, pts_2d_n, M_opt);
ave_res = mean(res_test);

%% 1-c
Q = M_norm(:, 1:3);
m4 = M_norm(:, 4);
C = -inv(Q) * m4;






