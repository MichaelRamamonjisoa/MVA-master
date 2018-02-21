[x,Fs] = audioread('Sons/guitare.wav');
% soundsc(x,Fs);
x = x';

n_vec = 0:1:length(x)-1;


% Phasing
f_a = 10;
amax = 2;
amin = 0;
a = ((amin+amax)/2)+((amax-amin)/2)*sin(n_vec*2*f_a*pi/Fs);

figure(); 
plot(n_vec, a);

p = 1000; %delay

y = x(p:end) + a(p:end).*x(1:end-p+1);

soundsc(y,Fs);
fprintf('Press any key to continue')
pause

% Flanger

a = 1.1;
f_p = 5;
pmax = 100;
pmin = -10;
p = ((pmax+pmin)/2)+((pmax-pmin)/2)*sin(n_vec*2*f_p*pi/Fs);

y = zeros(1,length(x));
for n=1:length(x)
    y(n) = x(n) + a*x(max(1,n-floor(p(n))+1));
end

soundsc(y,Fs);

pmax = floor(Fs/200);
pmin = floor(Fs/400);

p = ((pmax+pmin)/2)+((pmax-pmin)/2)*sin(n_vec*2*f_p*pi/Fs);

y = zeros(1,length(x));
for n=1:length(x)
    y(n) = x(n) + a*x(max(1,n-floor(p(n))+1));
end

soundsc(y,Fs);

% Artificial reverberation
c = 340 ;


% Schroeder reverberator
delay_comb = [29.7e-3, 37.1e-3, 41.4e-3, 43.7e-3];
delay_allpass = [96.83e-3, 32.92e-3];

T = 1/Fs;
Tr = 0.5; %release time of the reverb

m_comb = floor(delay_comb/T);
m_allpass = floor(delay_allpass/T);

g_comb = 10.^(-3*m_comb*(T/Tr));

g_allpass = exp(m_allpass*log(1-7*T/Tr));

y_out = computeReverb(x, m_comb, g_comb, m_allpass, g_allpass);
soundsc(y_out,Fs);

