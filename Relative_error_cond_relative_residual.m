A = [3 2
    1 1];
b = [5 2]';
x_hat = [1.1 0.88]';
x = [1 1]';


% Cond
cond(A,inf)

% Relative Error
re = norm(x-x_hat,'inf')/norm(x,'inf')

% Relaive Residual
rr = norm(b-A*x_hat,inf)/norm(b,inf)