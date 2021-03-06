problem_3a;
y1 = hist_obj;
y_size = size(y1);
x1 = [1:y_size(2)];

problem_3b;
y2 = hist_obj;
y_size = size(y2);
x2 = [1:y_size(2)];

problem_3c;
y3 = hist_obj;
y_size = size(y3);
x3 = [1:y_size(2)];

plot(x1, y1, x2, y2, x3, y3);
title('Problem 3 results');
xlabel('Number of iterations');
ylabel('Object function values');
legend({'3(a)','3(b)', '3(c)'}, 'Location','northeast')