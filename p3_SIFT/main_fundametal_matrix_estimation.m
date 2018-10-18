%% PS3 part2
close all; clear; clc;
% We now wish to estimate the mapping of 
% points in one image to lines in another by means of the fundamental matrix
%% 2-a
% Read points
tmp_2d_a = textread(fullfile('input', 'pts2d-pic_a.txt'), '%f');
tmp_2d_b = textread(fullfile('input', 'pts2d-pic_b.txt'), '%f');

n_pts = length(tmp_2d_a) / 2;
pts_2d_a_homo = zeros(3, n_pts);
pts_2d_b_homo = zeros(3, n_pts);

for i = 1 : n_pts
    pts_2d_a_homo(:, i) = [tmp_2d_a(2*i-1); tmp_2d_a(2*i); 1];
    pts_2d_b_homo(:, i) = [tmp_2d_b(2*i-1); tmp_2d_b(2*i); 1];
end


% Compute A; Left image = image a; Right image = image b

% A = zeros(n_pts, 9);
% for i = 1 : n_pts
%     ua = pts_2d_a_homo(1, i);
%     va = pts_2d_a_homo(2, i);
%     ub = pts_2d_b_homo(1, i);
%     vb = pts_2d_b_homo(2, i);
%     A(i, :) = [ub*ua, ub*va, ub, vb*ua, vb*va, vb, ua, va, 1];
% end

A = zeros(n_pts, 9);
for i = 1 : n_pts
    tmp = pts_2d_a_homo(:, i) * pts_2d_b_homo(:, i)';
    A(i, :) = tmp(:)';
end

%SVD to get F
[U1, D1, V1] = svd(A, 0);
f = V1(:, end);

%% Construct matrix F
F = zeros(3, 3);
for i = 1 : 3
    F(i, :) = f(3*i-2 : 3*i)';
end

%% 2-b
[U, D, V] = svd(F, 0);

D_ = D;
D_(end) = 0;

F_ = U * D_ * V';

%% 2-c
im_a = imread(fullfile('input', 'pic_a.jpg'));
im_b = imread(fullfile('input', 'pic_b.jpg'));
[hei, wid, dep] = size(im_a);

% epipolar lines
eplines_b = F_ * pts_2d_a_homo;
eplines_a = F_' * pts_2d_b_homo;

% line_L and line_R are in the same image.
line_L = cross([1; 1; 1], [hei; 1; 1]);
line_R = cross([1; wid; 1], [hei; wid; 1]);

n_eplines = n_pts; % Trival; Just for future reference

figure, imshow(im_b);
hold on
for i = 1 : n_eplines
    pts_L = cross(eplines_b(:, i), line_L);
    pts_R = cross(eplines_b(:, i), line_R);

    pts_L_ = pts_L(1:2) / pts_L(3);
    pts_R_ = pts_R(1:2) / pts_R(3);

    line([pts_L_(1), pts_R_(1)], [pts_L_(2), pts_R_(2)], ...
                        'Color', 'green', 'LineWidth', 1);
    
end
scatter(pts_2d_b_homo(1, :), pts_2d_b_homo(2, :), 12, 'r', 'filled');
hold off
title('image a');

figure, imshow(im_a);
hold on
for i = 1 : n_eplines
    pts_L = cross(eplines_a(:, i), line_L);
    pts_R = cross(eplines_a(:, i), line_R);

    pts_L_ = pts_L(1:2) / pts_L(3);
    pts_R_ = pts_R(1:2) / pts_R(3);

    line([pts_L_(1), pts_R_(1)], [pts_L_(2), pts_R_(2)], ...
                        'Color', 'green', 'LineWidth', 1);
end
scatter(pts_2d_a_homo(1, :), pts_2d_a_homo(2, :), 12, 'r', 'filled');
hold off
title('image b');

%% 2-d
[pts_2d_a_n, T_a] = pts_normalization(pts_2d_a_homo);
[pts_2d_b_n, T_b] = pts_normalization(pts_2d_b_homo);

A = zeros(n_pts, 9);
for i = 1 : n_pts
    tmp = pts_2d_a_n(:, i) * pts_2d_b_n(:, i)';
    A(i, :) = tmp(:)';
end

[U1, D1, V1] = svd(A, 0);
f = V1(:, end);
F = zeros(3, 3);
for i = 1 : 3
    F(i, :) = f(3*i-2 : 3*i)';
end
[U, D, V] = svd(F, 0);
D_ = D;
D_(end) = 0;
F_ = U * D_ * V';

%% 2-e
F_new = T_b' * F_ * T_a;
% epipolar lines
eplines_b = F_new * pts_2d_a_homo;
eplines_a = F_new' * pts_2d_b_homo;

% line_L and line_R are in the same image.
line_L = cross([1; 1; 1], [hei; 1; 1]);
line_R = cross([1; wid; 1], [hei; wid; 1]);

figure, imshow(im_b);
hold on
for i = 1 : n_eplines
    pts_L = cross(eplines_b(:, i), line_L);
    pts_R = cross(eplines_b(:, i), line_R);

    pts_L_ = pts_L(1:2) / pts_L(3);
    pts_R_ = pts_R(1:2) / pts_R(3);

    line([pts_L_(1), pts_R_(1)], [pts_L_(2), pts_R_(2)], ...
                        'Color', 'green', 'LineWidth', 1);
end
scatter(pts_2d_b_homo(1, :), pts_2d_b_homo(2, :), 10, 'r', 'filled');
hold off
title('image a with better estimated F');

figure, imshow(im_a);
hold on
for i = 1 : n_eplines
    pts_L = cross(eplines_a(:, i), line_L);
    pts_R = cross(eplines_a(:, i), line_R);

    pts_L_ = pts_L(1:2) / pts_L(3);
    pts_R_ = pts_R(1:2) / pts_R(3);

    line([pts_L_(1), pts_R_(1)], [pts_L_(2), pts_R_(2)], ...
                        'Color', 'green', 'LineWidth', 1);
end
scatter(pts_2d_a_homo(1, :), pts_2d_a_homo(2, :), 10, 'r', 'filled');
hold off
title('image b with better estimated F');


