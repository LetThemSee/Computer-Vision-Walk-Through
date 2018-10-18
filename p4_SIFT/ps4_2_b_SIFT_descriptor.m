function [F_out, D_out] = ps4_2_b_SIFT_descriptor(img, keypoints, angles)

n_points = size(keypoints, 1);
F_in = zeros(4, n_points);
for i = 1 : n_points
    F_in(1:2, i) = flip(keypoints(i, :));
    F_in(3, i) = 1;
    F_in(4, i) = angles(i);
end

[F_out, D_out] = vl_sift(single(img), 'frames', F_in);
