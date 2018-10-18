function plot_matches(pair, F_outA, F_outB, matches, imgA)
xA = F_outA(1, matches(1, :));
yA = F_outA(2, matches(1, :));

xB = F_outB(1, matches(2,:)) + size(imgA, 2);
yB = F_outB(2, matches(2,:));

figure, imshow(uint8(pair));
hold on ;
h = line([xA ; xB], [yA ; yB]) ;
set(h, 'linewidth', 1, 'color', 'b') ;

vl_plotframe(F_outA(:, matches(1, :)));
F_outB(1, :) = F_outB(1, :) + size(imgA, 2);
vl_plotframe(F_outB(:, matches(2, :)) );