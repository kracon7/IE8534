function [beta_next] = prox(lambda, t, beta_prev)
    beta_next = beta_prev;
    beta_next(beta_prev > lambda * t) = beta_prev(beta_prev > lambda * t) - lambda * t;
    beta_next(beta_prev < -lambda * t) = beta_prev(beta_prev < -lambda * t) + lambda * t;
    beta_next(abs(beta_prev) <= lambda * t) = 0;
end