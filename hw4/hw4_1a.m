clear;
rng('default');
M = zeros(20, 10);
M (1 : 3, :) = rand(3, 10);
r = rand(17, 3);
for i = 1 : 17
    M (i + 3, :) = (M (1, :) * r(i, 1) + M (2, :) * r(i, 2) + M (3, :) * r(i, 3))/sum(r(i, :));
end

I = zeros(120, 1); J = zeros(120, 1);
r = randperm(200, 120);
for k = 1 : 120
    i = ceil(r(k)/10); j = mod(r(k), 10);
    if (j == 0); j = 10; end
    I(k) = i; J(k) = j;
end

theta = 12;
L = 1;
t = 1 / L;
nStep = 0;
X = zeros(20, 10);

% initialize parapeters
hist_obj = [];
hist_nuc_norm = [];
hist_error = [];
time_start = cputime;

while 1
    % compute loss, gradient, nuclear norm
    nuc_norm = norm(svd(X), 1);
    [loss, grad] = objective(X, M, I, J);
    
    X = prox_indicator(X - t * grad, theta);
    
    % compute error for termination criterion
    if mod(nStep, 10) == 0
        [~, grad] = objective(X, M, I, J);
        error = termination(X, grad, theta);
        hist_error = [hist_error, error];
        
        if error <= 1e-3
            break
        end
    end
    
    nStep = nStep + 1;

    hist_obj = [hist_obj, loss];
    hist_nuc_norm = [hist_nuc_norm, nuc_norm];
end

fprintf('HW4 Problem 1(a). Total computation time is: %.5fs\n',cputime - time_start);
fprintf('Number of iterations is %d, \n', nStep);
fprintf('Final objective value is %.4f\n', loss);

subplot(1,3,1);
hist_size = size(hist_obj);
plot([1:hist_size(2)], hist_obj);
title('HW 4 Problem 1(a) result');
xlabel('Number of iterations');
ylabel('Objective function values');

subplot(1,3,2); 
plot([1:hist_size(2)], hist_nuc_norm);
xlabel('Number of iterations');
ylabel('Nuclear norm');

subplot(1,3,3);
hist_size = size(hist_error);
plot([1:hist_size(2)], hist_error);
xlabel('Number of iterations');
ylabel('Error');


function [error] = termination(X, grad, theta)
    sigma_max = svds(grad, 1);
    error = (trace(grad' * X) + theta * sigma_max) / max(1, max(grad, [], 'all'));
end

function [loss, grad] = objective(X, M, I,J)
    loss = 0.5 * norm(diag(X(I, J)) - diag(M(I, J))).^2;
    grad = zeros(size(X));
    index_size = size(I);
    for n = 1:index_size(1)
        grad(I(n), J(n)) = X(I(n), J(n)) - M(I(n), J(n));
    end
end

function [X] = prox_indicator(Y, theta)
    % proximal gradient of indicator function
    [U,D,V] = svd(Y);   % D: 20x10
    d = diag(D);           % d: 10
    d_star = ProjectOntoL1Ball(d, theta);   % d_star: 10
    D_star = zeros(size(D));  % D_star: 20x10
    D_star(1:10, :) = diag(d_star);
    X = U * D_star * V';
end