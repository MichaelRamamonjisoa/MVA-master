function y = computeEffectOutput(x,a,p)

N = length(x);
y = zeros(1,N);
for n=(1+p):N
    y(n) = x(n-p) + a*y(n-p);
end
