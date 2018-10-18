function opt_conse = compute_trans_consensus(N, toler, F_outA, F_outB, matches)
n_matches = size(matches, 2);
n_conse_set = zeros(N, 1);
conse_set = cell(N, 1);
for iter = 1 : N
    r = randi([1 n_matches]);
    sample = matches(:, r);
    offset_x = F_outB(1, sample(2)) - F_outA(1, sample(1));
    offset_y = F_outB(2, sample(2)) - F_outA(2, sample(1));

    consensus = [];
    for i = 1 : n_matches
        if i == r
            continue
        end
        mat = matches(:, i);
        tmp_x = F_outB(1, mat(2)) - F_outA(1, mat(1));
        tmp_y = F_outB(2, mat(2)) - F_outA(2, mat(1));

        if abs(tmp_x - offset_x) < toler && abs(tmp_y - offset_y) < toler
            consensus = [consensus, mat];
        end
    end

    conse_set{iter} = consensus;
    n_conse_set(iter) = size(consensus, 2);

end

opt_idx = find(n_conse_set == max(n_conse_set));
opt_conse = conse_set{opt_idx(1)};
