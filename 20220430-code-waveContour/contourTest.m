figure(2);
x = -2:0.2:2;
y = -2:0.2:3;
[X,Y] = meshgrid(x,y);
Z = 2.*X.*exp(-X.^2-Y.^2);
figure(1);
contour(X,Y,'ShowText','on')
figure(2);
contour(X,Y,Z,'ShowText','on')