function [r, J] = rosenbrockv(x, p1,p2)
if p2 > 0, r = [p1*(x(2) - x(1)^2); 1 - x(1); sqrt(2*p2)];
else, r = [p1*(x(2) - x(1)^2); 1 - x(1)]; end
if nargout > 1 % also the Jacobian
if p2 > 0, J = [-2*p1*x(1) p1; -1 0; 0 0];
else, J = [-2*p1*x(1) p1; -1 0]; end
end
