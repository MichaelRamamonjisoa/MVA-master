function y = computeEarlyReverbs(x,S,M,X,Y,Z,Fs)

c = 340;
N = length(x);

D = zeros(1,6);
xs = S(1);
ys = S(2);
zs = S(3);
xm = M(1);
ym = M(2);
zm = M(3);

D(1) = sqrt((xs+xm)^2+(ys-ym)^2+(zs-zm)^2);
D(2) = sqrt((xs-xm)^2+(ys+ym)^2+(zs-zm)^2);
D(3) = sqrt((xs-xm)^2+(ys-ym)^2+(zs+zm)^2);
D(4) = sqrt((2*X-(xs+xm))^2+(ys-ym)^2+(zs-zm)^2);
D(5) = sqrt((xs-xm)^2+(2*Y-(ys+ym))^2+(zs-zm)^2);
D(6) = sqrt((xs-xm)^2+(ys-ym)^2+(2*Z-(zs+zm))^2);
D(7) = sqrt((xs-xm)^2+(ys-ym)^2+(zs-zm)^2);
y = zeros(1,N);
b = []

y_temp = zeros(1,N);
for i=1:7
    a = min(1/D(i),1);
    y_temp = computeEffectOutput(x,a,10*floor(D(i)/c)*Fs); 
    plot(y_temp);
    y = y+ y_temp;
end
end