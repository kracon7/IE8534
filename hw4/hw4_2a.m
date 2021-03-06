rng('default');
A = randn(500, 50);
b = randn(500, 1);
lambda = 0.01 * max(A' * b);
mu = 0.005 * max(A' * b);
gamma = 0.001 * max(A' * b);
K = 5;
n = 50;
group = 5;

x = zeros(50, 1);

nStep = 2000;
x_optimal = x;
obj_optimal = objective(x, A, b, lambda, mu, gamma);
obj_hist = [obj_optimal];

for k = 1:nStep
    x = x - 1/k * get_grad(x, A, b, lambda, mu, gamma);
    obj = objective(x, A, b, lambda, mu, gamma);
    obj_hist = [obj_hist, obj];
    if obj < obj_optimal
        x_optimal = x;
        obj_optimal = obj;
    end
end

fprintf('HW4 Problem 2(a). Smallest objective function value found is: %.5f\n',obj_optimal);
hist_size = size(obj_hist);
semilogx([1:hist_size(2)], obj_hist);
title('HW4 2(a) result');
xlabel('Number of iterations');
ylabel('Object function values');

function [grad] = get_grad(x, A, b, lambda, mu, gamma)
    if any(A*x-b)
        grad = A' * (A*x-b) / norm(A*x-b);
    else
%         grad = 2 * rand(size(x)) - 1;
        grad = zeros(size(x));
    end
    
    l1_grad = lambda * sign(x);
    
    group_grad = mu * [l2_grad(x(1:10)) ; l2_grad(x(11:20)) ; l2_grad(x(21:30)) ;...
                 l2_grad(x(31:40)) ; l2_grad(x(41:50))];
    
    fused_grad = zeros(size(x));
    for i = 1:1:49
        if x(i) - x(i+1) ~= 0
            fused_grad(i) =  fused_grad(i) + gamma * sign(x(i) - x(i+1));
            fused_grad(i+1) = fused_grad(i+1) + gamma * sign(x(i) - x(i+1));
        end
    end
    
    grad = grad + l1_grad + group_grad + fused_grad;
       
end

function [grad] = l2_grad(x)
    if any(x)
        grad = x / norm(x, 2);
    else
%         grad = 2 * rand(size(x)) - 1;
        grad = zeros(size(x));        
    end
end

function [loss] = objective(x, A, b, lambda, mu, gamma)
    group_term = norm(x(1:10), 2) + norm(x(11:20), 2) + norm(x(21:30), 2) +...
                norm(x(31:40), 2) + norm(x(41:50), 2);
    fused_term = sum(x(1:49) - x(2:50));    
    loss = norm(A*x-b, 2) + lambda * norm(x,1) + mu * group_term + gamma * fused_term; 
end
