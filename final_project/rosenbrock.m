function [loss, grad] = rosenbrock(a, b, xx)
    x = xx(1);
    y = xx(2);
    loss = (a-x)^2 + b * (y - x^2)^2;
    grad = [2*(x-a)-4*b*x*(y-x^2), 2*b*(y-x^2)];
end