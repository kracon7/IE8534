% initialize parapeters
rng('default')
X = randn(1000, 200);
y = randn(1000, 1);
beta = zeros(200,1);
beta0 = 0.; 
mu = 1e-4;
lambda = max(0.5 * X' * (mean(y)-y));
Lip = max([eig(X' * X+ mu); 200+mu]);
% step size
t = 1 / Lip;
nStep = 0;

% initialize parapeters
hist_obj = objective(X, y, beta0, beta, mu, lambda);
time_start = cputime;
num_outer_iteration = 0;
num_prox_eval = 0;

while 1
    num_outer_iteration = num_outer_iteration + 1;
    
    % compute gradient
    grad_beta0 = (1000 + mu) * beta0 + sum((X * beta - y));
    grad_beta = X' * (beta0 + X * beta - y) + mu * beta;
    % step the gradient
    beta0 = beta0 - t * grad_beta0;
    beta= beta - t * grad_beta;
    beta = prox(lambda, t, beta);
    
    num_prox_eval = num_prox_eval + 1;
    
    % compute error for termination criterion
    temp_beta0 = beta0 - grad_beta0;
    temp_beta = prox(lambda, 1, beta - grad_beta);
    error = max(abs([temp_beta; temp_beta0] - [beta; beta0])) / max([1, objective(X, y, beta0, beta, mu, lambda)]);
    
    num_prox_eval = num_prox_eval + 1;    
    
    if error <= 1e-6 || nStep > 1e2
        break
    end
    
    nStep = nStep + 1;
    
    % compute objective function value and save
    loss = objective(X, y, beta0, beta, mu, lambda);
    hist_obj = [hist_obj, loss];
end

fprintf('Problem 4(a). Total computation time is: %.5fs\n',cputime - time_start);
fprintf('Number of outer iterations is %d, number of proximal gradient evaluation is %d\n', ...
            num_outer_iteration, num_prox_eval);
fprintf('Final objective value is %.4f\n', loss);

hist_size = size(hist_obj);
plot([1:hist_size(2)], hist_obj);
title('Problem 4(a) result');
xlabel('Number of iterations');
ylabel('Object function values');

function obj = objective(X, y, beta0, beta, mu, lambda)
    obj = 0.5* (beta0 + X * beta - y)' * (beta0 + X * beta - y) + 0.5 * mu *(beta0 ^ 2 + beta' * beta) + lambda * sum(abs(beta));
end