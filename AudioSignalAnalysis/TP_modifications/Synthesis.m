function y = Synthesis(s,Fs,A,B)

n_marks = size(A,2);
ts_kend = B(1,end);
Pa_nkend = A(3,B(2,end));

y = zeros(1,ts_kend + Pa_nkend);

for k=2:size(B,2)
    
    nk = B(2,k);
    ts = B(1,k);
    ta = A(1,nk);
    Pa = A(3,nk);
% extraction of x
    x = s(1,ta-Pa:ta+Pa);
    i_start = ta-Pa;
    i_end = ta+Pa;
    
% windowing of x    
    w = window('hann',length(x));
    x = x.*w';
    figure(1)
    plot(x);
% overlap-add

    x_OLA = x;
    ts = B(1,k);
    OL_duration = length(ts - Pa:ts + Pa);
    if OL_duration > length(x_OLA)
%         pad x_OLA
        n_pads = OL_duration - length(x_OLA);
        x_OLA = [zeros(1,floor(n_pads/2)),x_OLA,zeros(1,floor(n_pads/2))-1];
    else
        n_pads = length(x_OLA) - OL_duration;
        x_OLA = x_OLA(1+floor(n_pads/2):length(x_OLA)-floor(n_pads/2));
        
    end
    
    y(ts - Pa:ts+Pa) = y(ts - Pa:ts+Pa) + x_OLA;   
    
end
figure(2); plot(y)
end