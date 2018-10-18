function [R, R_norm] = ps4_1_b_Harris_value(X_grad, Y_grad, alpha, win_size)

[height, width] = size(X_grad);

window = ones(win_size);
half_win = round(win_size / 2);

R = zeros(height, width);
for i = half_win : height-half_win+1
    for j = half_win : width-half_win+1
        M = zeros(2);
        for offset_x = -half_win+1 : half_win-1
            for offset_y = -half_win+1 : half_win-1
                x = j + offset_x;
                y = i + offset_y;
                I_xx = X_grad(y, x) ^ 2; 
                I_yy = Y_grad(y, x) ^ 2; 
                I_xy = X_grad(y, x) * Y_grad(y, x);
                
                M = M + window(offset_y+half_win, offset_x+half_win) * [I_xx, I_xy; I_xy, I_yy];
            end
        end
        R(i, j) = det(M) - alpha * trace(M)^2;
    end
end

R_norm = ( R - min(R(:)) ) ./ ( max(R(:)) - min(R(:)) ) * 255 + 1;


