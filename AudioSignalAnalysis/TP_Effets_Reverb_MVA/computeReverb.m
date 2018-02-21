function y_out = computeReverb(x, m_comb, g_comb, m_allpass, g_allpass)
if length([m_comb,m_allpass])~=length([g_comb,g_allpass])
    fprintf('m and g not of the same length')
end

y_out_comb = computeCombFilterOutput(x,m_comb, g_comb);
y_out = computeAllPassOutput(y_out_comb,m_allpass, g_allpass);

end

function y = computeCombFilterOutput(x, m, g)
figure(3)
hold on;
N = length(x);

y = zeros(1,N);
for i=1:length(m)
% %     y(n) = x(n-m) + g*y(n-m)
%     y_temp = zeros(1,N);
%     for n=(1+m(i)):N
% %     y_temp = x + g(i)*[x(m(i):end),zeros(1,N-length(x(m(i):end)))];
% %     y = y + y_temp;
% % How to initialize the y?
%        y_temp(n) = x(n-m(i)) + g(i)*y_temp(n-m(i));
%     end
%     plot(y_temp);
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