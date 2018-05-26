function A = AnalysisPitchMarks(s,Fs)

ta = 1;
Pa = 0.01*Fs;

N = length(s);

A = zeros(3,floor(N/2));
A(1,1) = ta;
[~, A(2,1)] = periode(s(1:floor(2.5*0.01*Fs)),Fs);
A(3,1) = Pa;

n_max = 2.5*Pa;
n=1;
while n_max<N
    
    x_start = ta;
    x_duration = 2.5*Pa;
    x = s(x_start:x_start + floor(x_duration));
    
    [Pa,A(2,n)] = periode(x,Fs);
    
    ta = ta + Pa;
    
    A(3,n) = Pa; 
    A(1,n) = ta;
    
    n = n+1;
    n_max = ta+floor(2.5*Pa);
end

A = A(1:end,1:n-1);
end