clear all;
syms x1 x2
f_1 = @(x1,x2) (x1^2)-(x2^2);
f_2 = @(x1,x2) 2*x1*x2;
figure(1)
zhandle = fcontour(f_1)
hold on
zhandle2 = fcontour(f_2)


zhandle.LevelList = 12;
zhandle2.LevelList = 16;
xlabel x1
ylabel x2
title('Dank Memes')
grid on
axis equal

% The crossing can be seen to be in (4,2) and (-4,-2)

%% Find crossings
clear all;
syms x1 x2
f_1 = @(x1,x2) (x1^2)-(x2^2) == 12;
f_2 = @(x1,x2) 2*x1*x2 == 16;

x_2 = solve(f_2,x2) % 8/x1

% Input x_2 into f_1
f_1 = @(x1) (x1^2)-((8/x1)^2) == 12;

x1 = real(solve(f_1,x1))

%For x1 = -4
f_2_fin = @(x2) 2*(-4)*x2 == 16;
x2_min4 = solve(f_2_fin,x2)

%For x1 = 4
f_2_fin = @(x2) 2*4*x2 == 16;
x2_4 = solve(f_2_fin,x2)