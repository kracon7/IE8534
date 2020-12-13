clear;
x = linspace(-10,10,100);
y = linspace(-10,10,100);
[X,Y] = meshgrid(x,y);
Z = X.^2 - Y.^2;

x = [5.5,0.01];
t = 1e-2;
nStep = 2000;
[obj_optimal, ~] = saddle(x);
obj_hist = [obj_optimal];
x_hist = [x];

s_k = zeros(2,2);

for k = 1:nStep
    [obj, grad] = saddle(x);
    s_k = s_k + 0.01*diag(grad.^2);
    x = x - t * grad *inv(sqrt(s_k));
    
    obj_hist = [obj_hist, obj];
    x_hist = [x_hist; x];
    
    
    % if saddle value is smaller than -1e3, stop optimization
    if obj < -1e3
        break
    end
end

fprintf('AdaGrad method. The smallest saddle function value found is: %.5f\n',obj_optimal);
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