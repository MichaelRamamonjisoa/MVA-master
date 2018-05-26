function y_out = computeReverb(x, Tr, Fs)

delay_comb = [29.7e-3, 37.1e-3, 41.4e-3, 43.7e-3];
delay_allpass = [96.83e-3, 32.92e-3];

T = 1/Fs;

m_comb = floor(delay_comb/T);
m_allpass = floor(delay_allpass/T);

g_comb = 10.^(-3*m_comb*(T/Tr));

g_allpass = exp(m_allpass*log(1-7*T/Tr));

y_out_comb = computeCombFilterOutput(x,m_comb, g_comb);
y_out = computeAllPassOutput(y_out_comb,m_allpass, g_allpass);

end

function y = computeCombFilterOutput(x, m, g)
figure()
hold on;
N = length(x);

y = zeros(1,N);
for i=1:length(m)
    y_temp = computeEffectOutput(x,g(i),m(i));
    y = y + y_temp;
    plot(y_temp);
end
hold off
end

function y = computeAllPassOutput(x,m,g)

N = length(x);
y = zeros(1,N);

x_int = zeros(1,N);
for n=1+m(1):N
    x_int(n) = x(n-m(1)) + g(1)*(x_int(n-m(1))-x(n));
end


for i=2:length(m)
    y_temp = zeros(1,N);
    for n=(1+m(i)):N
        y_temp(n) = x_int(n-m(i)) + g(i)*(y_temp(n-m(i)) - x_int(n));
    end
    x_int = y_temp;
end

y = x_int;

end