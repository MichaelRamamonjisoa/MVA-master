clear all; close all

[x,Fs] = audioread('Sons/guitare.wav');
% soundsc(x,Fs);
x = x';

n_vec = 0:1:length(x)-1;


% Phasing
f_a = 5;
amax = 1.5;
amin = 0.5;
a = ((amin+amax)/2)+((amax-amin)/2)*sin(n_vec*2*f_a*pi/Fs);

figure(); 
plot(n_vec, a);

p = 1000; %delay

y = x(p:end) + a(p:end).*x(1:end-p+1);

soundsc(y,Fs);
fprintf('Reading Phasing Effect. Press any key to continue \n')
audiowrite('guitare_Phasing.wav',y,Fs);
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
fprintf('Reading Flanger Effect. Press any key to continue \n')
pause


pmax = floor(Fs/200);
pmin = floor(Fs/400);

p = ((pmax+pmin)/2)+((pmax-pmin)/2)*sin(n_vec*2*f_p*pi/Fs);

y = zeros(1,length(x));
for n=1:length(x)
    y(n) = x(n) + a*x(max(1,n-floor(p(n))+1));
end

soundsc(y,Fs);
fprintf('Cancelled f = 200Hz, Press any key to continue \n')
audiowrite('guitare_Flanger.wav',y,Fs);
pause

% Artificial reverberation
c = 340 ;
% Early reverbs
S = [8,8,3];
M = [13,20,4];
X = 50; Y = 50; Z = 50;
y_Early = computeEarlyReverbs(x,S,M,X,Y,Z,Fs);
soundsc(y_Early, Fs);
fprintf('Press any key to continue \n')
audiowrite('guitare_EarlyReverb.wav',y_Early,Fs);
pause

% Schroeder reverberator

T = 1/Fs;
Tr = 0.5; %release time of the reverb

y_out = computeReverb(x, Tr,Fs);
audiowrite('guitare_Reverb.wav',y_out,Fs);
soundsc(y_out,Fs);