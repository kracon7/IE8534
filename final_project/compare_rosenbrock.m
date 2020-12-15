clear;
linx = linspace(-3,3,100);
liny = linspace(-2,8,100);
[X,Y] = meshgrid(linx,liny);
a = 1;
b = 1;
Z = (a-X).^2 + b * (Y - X.^2).^2;
contour3(X,Y,Z,[-5:1:200]);
xlabel('x');
ylabel('y');
zlabel('objective value');
view(135, 60)

hold on

t = 1e-2;
nStep = 400;

x0 = [-1, 5];
[obj, grad] = rosenbrock(a, b, x0);
v_prev_hb = zeros(size(x0));
v_prev_nag = zeros(size(x0));
s_k = zeros(2,2);
gamma = 0.9;
min_thresh = 1e-3;

% initialize x
x_sgd = x0;
x_hb =  x0;
x_nag =  x0;
x_ada =  x0;

% initialize objective value and x log
obj_hist_sgd = [obj];
obj_hist_hb = [obj];
obj_hist_nag = [obj];
obj_hist_ada = [obj];
x_hist_sgd = [x0];
x_hist_hb = [x0];
x_hist_nag = [x0];
x_hist_ada = [x0];
grad_sgd = grad;
grad_hb = grad;
grad_nag = grad;
grad_ada = grad;

for k = 1:nStep
    % gradient descent
    x_sgd = x_sgd - t * grad_sgd;
    [obj_sgd, grad_sgd] = rosenbrock(a, b, x_sgd);
    if obj_sgd > min_thresh
        obj_hist_sgd = [obj_hist_sgd, obj_sgd];
        x_hist_sgd = [x_hist_sgd; x_sgd];
        p1 = plot3(x_hist_sgd(:,1), x_hist_sgd(:,2), obj_hist_sgd, 'r','LineWidth',2);
        hold on
    end
    
    
    
    % heavy ball 
    v_hb = gamma * v_prev_hb + t * grad_hb;
    x_hb = x_hb - v_hb;
    v_prev_hb = v_hb;
    
    [obj_hb, grad_hb] = rosenbrock(a, b, x_hb);
    if obj_hb > min_thresh
        obj_hist_hb = [obj_hist_hb, obj_hb];
        x_hist_hb = [x_hist_hb; x_hb];
        p2 = plot3(x_hist_hb(:,1), x_hist_hb(:,2), obj_hist_hb, 'g','LineWidth',2);
        hold on
    end
        
    % nestrov accelarated gradient
    [~, next_grad] = rosenbrock(a, b, x_nag - gamma * v_prev_nag);
    v_nag = gamma * v_prev_nag + t * next_grad;
    x_nag = x_nag - v_nag;
    v_prev_nag = v_nag;
    [obj_nag, grad_nag] = rosenbrock(a, b, x_nag);
    if obj_nag > min_thresh
        obj_hist_nag = [obj_hist_nag, obj_nag];
        x_hist_nag = [x_hist_nag; x_nag];
        p3 = plot3(x_hist_nag(:,1), x_hist_nag(:,2), obj_hist_nag, 'b','LineWidth',2);
        hold on
    end 
    
    % adaptive gradient
    s_k = s_k + 0.01*diag(grad_ada.^2);
    x_ada = x_ada - t * grad_ada *inv(sqrt(s_k)+1e-8*eye(2));
    [obj_ada, grad_ada] = rosenbrock(a, b, x_ada);
    if obj_ada > min_thresh
        obj_hist_ada = [obj_hist_ada, obj_ada];
        x_hist_ada = [x_hist_ada; x_ada];
        p4 = plot3(x_hist_ada(:,1), x_hist_ada(:,2), obj_hist_ada, 'k','LineWidth',2);
        hold on
    end
    
    if obj_sgd < min_thresh && obj_hb < min_thresh && obj_nag < min_thresh && obj_ada < min_thresh
        break
    end
    
    legend([p1 p2 p3 p4],{'SGD','Heavy ball', 'Nestrov', 'AdaGrad'}, 'Location','northeast')
    pause(0.05)
end


figure;
hist_size_sgd = size(obj_hist_sgd);
semilogx([1:hist_size_sgd(2)], obj_hist_sgd, 'r');
hold on
hist_size_hb = size(obj_hist_hb);
semilogx([1:hist_size_hb(2)], obj_hist_hb, 'g');
hold on
hist_size_nag = size(obj_hist_nag);
semilogx([1:hist_size_nag(2)], obj_hist_nag, 'b');
hold on
hist_size_ada = size(obj_hist_ada);
semilogx([1:hist_size_ada(2)], obj_hist_ada, 'k');

title('comparing results of various gradient methods');
xlabel('Number of iterations');
ylabel('Object function values');
legend({'SGD','Heavy ball', 'Nestrov', 'AdaGrad'}, 'Location','northeast')

fprintf('On Saddle function, total number of steps before termination are:\n%15s %15s %15s %15s\n%15d %15d %15d %15d\n',...
        'SGD','Heavy ball', 'Nestrov', 'AdaGrad',...
        hist_size_sgd(2), hist_size_hb(2), hist_size_nag(2), hist_size_ada(2));
