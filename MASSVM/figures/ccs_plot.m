x = linspace(-2, 2, 1000);
z = x.*(x>0) + 0.*(x<=0);

y = @(x,nu) 0.*(x<=0) + (-x.^2/nu.^2 + 2*x/nu).*(x>0).*(x<nu) + 1.*(x>=nu);
y1 = y(x,1);
y2 = y(x,0.5);
y3 = y(x,0.1);

figure;
plot(x, y1, 'k-');
hold on;
plot(x, y2, 'k--');
hold on;
plot(x, y3, 'k:');
hold on;
plot(x, z, 'k-.');
hold on;

xlabel('z');
ylabel('l_{ccs}(z)');
legend('\nu = 1', '\nu = 0.5', '\nu = 0.1', 'Hinge loss', 'FontSize', 10, 'Location', 'northwest');
axis([-1 2 0 1.5]);
grid on;
