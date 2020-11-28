% initialize parapeters
rng('default')
X = randn(1000, 200);
y = randn(1000, 1);
beta = zeros(200,1);
beta0 = 0.; 
mu = 1e-4;
lambda = max(0.5 * X' * (mean(y)-y));
Lip = max([eig(X' * X+ mu); 200+mu]);

% initialization for Nesterov's method
m = min([eig(X' * X+ mu); 200+mu]);
v = [beta0; beta];
gamma = mu;

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
    
    theta = max(roots([Lip, gamma-m, -gamma]));
    y_next = [beta0; beta]  + theta * gamma * (v - [beta0; beta]) / (gamma + m * theta);
    
    % compute gradient
    grad_beta0 = (1000 + mu) * y_next(1) + sum((X * y_next(2: 201) - y));
    grad_beta = X' * (y_next(1) + X * y_next(2: 201) - y) + mu * beta;
    % step the gradient
    beta0_next = y_next(1) - t * grad_beta0;
    beta_next = y_next(2:201) - t * grad_beta;
    beta_next = prox(lambda, t, beta_next);
    
    num_prox_eval = num_prox_eval + 1;
    
    v = [beta0_next; beta_next] + ([beta0_next; beta_next] - [beta0; beta])/theta;
    gamma = theta^2 / t;
    
    beta0 = beta0_next;
    beta = beta_next;
    
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

fprintf('Problem 4(c). Total computation time is: %.5fs\n',cputime - time_start);
fprintf('Number of outer iterations is %d, number of proximal gradient evaluation is %d\n', ...
            num_outer_iteration, num_prox_eval);
fprintf('Final objective value is %.4f\n', loss);

hist_size = size(hist_obj);
plot([1:hist_size(2)], hist_obj);
title('Problem 4(c) result');
xlabel('Number of iterations');
ylabel('Object function values');

function obj = objective(X, y, beta0, beta, mu, lambda)
    obj = 0.5* (beta0 + X * beta - y)' * (beta0 + X * beta - y) + 0.5 * mu *(beta0 ^ 2 + beta' * beta) + lambda * sum(abs(beta));
end