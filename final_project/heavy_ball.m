clear;
x = linspace(-10,10,100);
y = linspace(-10,10,100);
[X,Y] = meshgrid(x,y);
Z = X.^2 - Y.^2;

x = [5.5,0.01];
t = 1e-2;
nStep = 2000;
[obj, grad] = saddle(x);
obj_hist = [obj];
x_hist = [x];

v_prev = zeros(size(x));
gamma = 0.9;

for k = 1:nStep
    v = gamma * v_prev + t * grad;
    x = x - v;
    v_prev = v;
    
    [obj, grad] = saddle(x);
    obj_hist = [obj_hist, obj];
    x_hist = [x_hist; x];
    
    % if saddle value is smaller than -1e3, stop optimization
    if obj < -2e2
        break
    end
end

fprintf('Momentum method. The smallest saddle function value found is: %.5f\n',obj);
hist_size = size(obj_hist);
plot([1:hist_size(2)], obj_hist);
title('SGD result');
xlabel('Number of iterations');
ylabel('Object function values');

figure;
contour3(X,Y,Z,[-5e2:2:5e2])
xlabel('x');
ylabel('y');
hold on
plot3(x_hist(:,1), x_hist(:,2), obj_hist, 'r')
