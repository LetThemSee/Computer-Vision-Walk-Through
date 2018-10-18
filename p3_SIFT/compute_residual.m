function residual = compute_residual(pts_3d, pts_2d_test, M)
    % Input:
    %       pts_3d is in a form of (3 x n_pts)
    %       pts_2d_test is in a form of (2 x n_pts)
    % Output:
    %       residual (1 x n_pts)
            

    n_pts = size(pts_3d, 2);

    pts_3d_homo = [pts_3d; ones(1, n_pts)];
    pts_2d_inhomo = M * pts_3d_homo;

    pts_2d_homo = zeros(3, n_pts);

    for i = 1 : n_pts
        pts_2d_homo(:, i) = pts_2d_inhomo(:, i) / pts_2d_inhomo(3, i);
    end

    pts_2d = pts_2d_homo(1:2, :);

    % Calculate the square root of ssd (i.e. residual)
    diff = pts_2d_test - pts_2d;
    ssd = sum(diff.^2);
    residual = sqrt(ssd);
