function [loss, grad] = parabolic(xx)
    x = xx(1);
    y = xx(2);
    a = 15;
    loss = x^2 + a*y^2;
    grad = [2*x, a*2*y];
end