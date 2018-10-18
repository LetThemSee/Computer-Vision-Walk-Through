function [opt_conse, opt_affine] = compute_affine_consensus(N, toler, F_outA, F_outB, matches)
n_matches = size(matches, 2);

n_conse_set = zeros(N, 1);
conse_set = cell(N, 1);
affine_set = cell(N, 1);
for iter = 1 : N
    r = randi([1 n_matches], 1, 3);
    sample = matches(:, r);
    
    pt1_A = F_outA(1:2, sample(1, 1));
    pt1_B = F_outB(1:2, sample(2, 1));
    
    pt2_A = F_outA(1:2, sample(1, 2));
    pt2_B = F_outB(1:2, sample(2, 2));
    
    pt3_A = F_outA(1:2, sample(1, 3));
    pt3_B = F_outB(1:2, sample(2, 3));
    
    A =[pt1_A(1), pt1_A(2), 1, 0, 0, 0;
            0, 0, 0, pt1_A(1), pt1_A(2), 1;
                pt2_A(1), pt2_A(2), 1, 0, 0, 0;
                    0, 0, 0, pt2_A(1), pt2_A(2), 1;
                        pt3_A(1), pt3_A(2), 1, 0, 0, 0;
                            0, 0, 0, pt3_A(1), pt3_A(2), 1];

    b = [pt1_B(1); pt1_B(2); pt2_B(1); pt2_B(2); pt3_B(1); pt3_B(2)];
    affine = A \ b;
    affine_set{iter} = reshape(affine, [3,2])';
    
    consensus = [];
    for i = 1 : n_matches-2
        mat = matches(:, i:i+2);
        
        pt1_A_tmp = F_outA(1:2, mat(1, 1));
        pt1_B_tmp = F_outB(1:2, mat(2, 1));

        pt2_A_tmp = F_outA(1:2, mat(1, 2));
        pt2_B_tmp = F_outB(1:2, mat(2, 2));

        pt3_A_tmp = F_outA(1:2, mat(1, 3));
        pt3_B_tmp = F_outB(1:2, mat(2, 3));

        A_tmp =[pt1_A_tmp(1), pt1_A_tmp(2), 1, 0, 0, 0;
                0, 0, 0, pt1_A_tmp(1), pt1_A_tmp(2), 1;
                    pt2_A_tmp(1), pt2_A_tmp(2), 1, 0, 0, 0;
                        0, 0, 0, pt2_A_tmp(1), pt2_A_tmp(2), 1;
                            pt3_A_tmp(1), pt3_A_tmp(2), 1, 0, 0, 0;
                                0, 0, 0, pt3_A_tmp(1), pt3_A_tmp(2), 1];

        b_tmp = [pt1_B_tmp(1); pt1_B_tmp(2); pt2_B_tmp(1); ...
                pt2_B_tmp(2); pt3_B_tmp(1); pt3_B_tmp(2)];
        
        affine_tmp = A_tmp \ b_tmp;
        
        diff = abs(affine - affine_tmp);
        if sum(diff(:)) < toler
            consensus = [consensus, mat];
        end
    end
    conse_set{iter} = consensus;
    n_conse_set(iter) = size(consensus, 2);
end

opt_idx = find(n_conse_set == max(n_conse_set));
opt_conse = conse_set{opt_idx(1)};
opt_affine = affine_set{opt_idx(1)};
