function angles = ps4_2_a_Angles(keypoints, X_grad, Y_grad)
n_points = size(keypoints, 1);
angles = zeros(n_points, 1);
for i = 1 : n_points
    point = keypoints(i, :);
    angles(i) = atan2(Y_grad(point(1), point(2)), X_grad(point(1), point(2)));
end
