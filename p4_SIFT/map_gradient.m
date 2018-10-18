function grad_output = map_gradient(grad)

grad_output = zeros(size(grad));
for i = 1 : size(grad, 1)
    for j = 1 : size(grad, 2)
        if grad(i, j) > 0
            grad_output(i, j) = 255;
        elseif grad(i, j) < 0
                grad_output(i, j) = 0;
            else
                grad_output(i, j) = 128;
        end
    end
end


            