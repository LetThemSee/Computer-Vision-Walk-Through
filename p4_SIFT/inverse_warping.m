function warpedB = inverse_warping(imgA, imgB, sim)
% Warp image B back to image A

warpedB = zeros(size(imgA)); % warpedB is the destination image of the same size as A
    
for y = 1 : size(warpedB, 1)
    for x = 1 : size(warpedB, 2)
        posB = sim * [x; y; 1]; % Inverse warping
        if posB(1) >= 1 && posB(1) <= size(imgB, 2) ...
                    && posB(2) <= size(imgB, 1) && posB(2) >= 1 
            pix_pos1 = flip(floor(posB)'); % flip to [row, col] fashion
            pix_pos2 = flip([ceil(posB(1)), floor(posB(2))]);
            pix_pos3 = flip(ceil(posB)');
            pix_pos4 = flip([floor(posB(1)), ceil(posB(2))]);
            a = posB(1) - floor(posB(1));
            b = posB(2) - floor(posB(2));
            warpedB(y, x) = (1-a)*(1-b)*imgB(pix_pos1(1), pix_pos1(2)) ...
                                + a*(1-b)*imgB(pix_pos2(1), pix_pos2(2)) ...
                                    + a*b*imgB(pix_pos3(1), pix_pos3(2)) ...
                                     + (1-a)*b*imgB(pix_pos4(1), pix_pos4(2));
        end
    end
end