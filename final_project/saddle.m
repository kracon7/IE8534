function [loss, grad] = saddle(xx)
    x = xx(1);
    y = xx(2);
    loss = x^2 - y^2;
    grad = [2*x, -2*y];
end