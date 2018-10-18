function plot_gradient(img, keypoints, angles)

%% Draw with VL

% f = zeros(4, n_points);
% for i = 1 : n_points
%     f(1:2, i) = flip(keypoints(i, :));
%     f(3, i) = 1;
%     f(4, i) = angles(i);
% end
% 
% figure, imshow(uint8(img));
% 
% h1 = vl_plotframe(f(:, 1:n_points)) ;
% h2 = vl_plotframe(f(:, 1:n_points)) ;
% set(h1,'color','k','linewidth',3) ;
% set(h2,'color','y','linewidth',2) ;
%% Draw 

line = 10; % Length of line that shows the direction of the gradient
n_points = size(keypoints, 1);

figure, imshow(uint8(img));
hold on
for i = 1 : n_points
    y = keypoints(i, 1);
    x = keypoints(i, 2);
    ori = angles(i);

    x_start = min(round(x - line * cos(ori) / 2), round(x + line * cos(ori) / 2));
    y_start = min(round(y - line * sin(ori) / 2), round(y + line * sin(ori) / 2));

    x_end = max(round(x - line * cos(ori) / 2), round(x + line * cos(ori) / 2));
    y_end = max(round(y - line * sin(ori) / 2), round(y + line * sin(ori) / 2));
    
    plot([x_start, x_end], [y_start, y_end], 'r', 'LineWidth', 1);
end
hold off