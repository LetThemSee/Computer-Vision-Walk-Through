function M = DLC_homo(pts_2d, pts_3d)
    % pts_2d is in a form of (2 x n_pts)
    n_pts = size(pts_2d, 2);
    
    % Construct matrix A
    A = zeros(2*n_pts, 12);

    for i = 1 : n_pts
        A(2*i-1, 1:4) = [pts_3d(:, i)', 1];
        A(2*i-1, 9:12) = -pts_2d(1, i) * [pts_3d(:, i)', 1];

        A(2*i, 5:8) = [pts_3d(:, i)', 1];
        A(2*i, 9:12) = -pts_2d(2, i) * [pts_3d(:, i)', 1];
    end

    % Compute m
    [U, D, V] = svd(A, 0);
    m = V(:, end);

    % Construct matrix M
    M = zeros(3, 4);
    for i = 1 : 3
        M(i, :) = m(4*i-3 : 4*i)';
    end

end
