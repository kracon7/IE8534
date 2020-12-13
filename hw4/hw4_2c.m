rng('default');
A = randn(500, 50);
b = randn(500, 1);
lambda = 0.01 * max(A' * b);
mu = 0.005 * max(A' * b);
gamma = 0.001 * max(A' * b);
beta = 0.25;
K = 5;
n = 50;
group = 5;

% initialize x
x = zeros(50, 1);

nStep = 2000;
x_optimal = x;
obj = objective(x, A, b, lambda, mu, gamma);
obj_optimal = obj;
obj_hist = [obj_optimal];
s_k = get_grad(x, A, b, lambda, mu, gamma);

% optimization begins
for k = 1:nStep
    
    grad = get_grad(x, A, b, lambda, mu, gamma);
    s_k = (1-beta) * grad + beta * s_k;
    step_size = (obj - obj_optimal + 10/(10+k)) / norm(s_k)^2;
    
    x = x - step_size * s_k;
    obj = objective(x, A, b, lambda, mu, gamma);
    obj_hist = [obj_hist, obj];
    if obj < obj_optimal
        x_optimal = x;
        obj_optimal = obj;
    end
end

fprintf('HW4 Problem 2(c). Smallest objective function value found is: %.5f\n',obj_optimal);
hist_size = size(obj_hist);
semilogx([1:hist_size(2)], obj_hist);
title('HW4 2(c) result');
xlabel('Number of iterations');
ylabel('Objective function values');

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
