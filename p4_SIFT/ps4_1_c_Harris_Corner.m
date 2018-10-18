function keypoints = ps4_1_c_Harris_Corner(R, threshold_factor, hood_size)
[height, width] = size(R); 
% Threshold out 
R_output = (R > threshold_factor*max(R(:))) .* R;
% Non-maximal suppression
half_hood_size = round(hood_size / 2);

%corners = zeros([height, width]);
keypoints = [];
for i = half_hood_size : height-half_hood_size+1
    for j = half_hood_size : width-half_hood_size+1
        row_start = i-half_hood_size+1;
        row_end = i+half_hood_size-1;
        col_start = j-half_hood_size+1;
        col_end = j+half_hood_size-1;
        hood = R_output(row_start:row_end, col_start:col_end);
        if  R_output(i, j) > 0 && R_output(i, j) == max(hood(:))
            %corners(i, j) = 1;
            keypoints = [keypoints; [i, j]];
        end
    end
end


