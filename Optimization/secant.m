function [x,v] = secant(g,xcurr,xnew,uncert)
%Matlab routine for finding root of g(x) using secant method
%
% secant;
% secant(g);
% secant(g,xcurr,xnew);
% secant(g,xcurr,xnew,uncert);
%
% x=secant;
% x=secant(g);
% x=secant(g,xcurr,xnew);
% x=secant(g,xcurr,xnew,uncert);
%
% [x,v]=secant;
% [x,v]=secant(g);
% [x,v]=secant(g,xcurr,xnew);
% [x,v]=secant(g,xcurr,xnew,uncert);
%
%The first variant finds the root of g(x) in the M-file g.m, with
%initial conditions 0 and 1, and uncertainty 10^(-5).
%The second variant finds the root of the function in the M-file specified
%by the string g, with initial conditions 0 and 1, and uncertainty 10^(-5).
%The third variant finds the root of the function in the M-file specified
%by the string g, with initial conditions specified by xcurr and xnew, and
%uncertainty 10^(-5).
%The fourth variant finds the root of the function in the M-file specified
%by the string g, with initial conditions specified by xcurr and xnew, and
%uncertainty specified by uncert.
%
%The next four variants returns the final value of the root as x.
%The last four variants returns the final value of the root as x, and
%the value of the function at the final value as v.
if nargin < 4
uncert=10^(-5);
if nargin < 3
if nargin == 1
xcurr=0;
xnew=1;
elseif nargin == 0
g="g";
else
disp("Cannot have 2 arguments."); % Determine either both intervals or none 
return;
end
end
end

g_curr=feval(g,xcurr);
while abs(xnew-xcurr)>xcurr*uncert % define how many digits should they be differ
xold=xcurr;
xcurr=xnew;
g_old=g_curr;
g_curr=feval(g,xcurr);
xnew=(g_curr*xold-g_old*xcurr)/(g_curr-g_old);
end %while
%print out solution and value of g(x)
if nargout >= 1
x=xnew;
if nargout == 2
v=feval(g,xnew);
end
else
final_point=xnew
value=feval(g,xnew)
end %if