function plot_corners(img, keypoints)

n_points = size(keypoints, 1);
%figure, imshow(uint8(img));
hold on
for i = 1 : n_points
    x = keypoints(i, 2);
    y = keypoints(i, 1);
    plot(x, y, 'r*', 'MarkerSize', 1);
end

hold off