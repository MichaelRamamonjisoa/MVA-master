function B = ChangeTimeScale(alpha,A,Fs)

n_marks = size(A,2);
ts = 1;
nk = 1;

B = zeros(2,n_marks);

k=1;

while nk<=n_marks
    
    Pa = A(3,floor(nk));
    
    B(1,k) = floor(ts);
    B(2,k) = floor(nk);
    
    ts = ts + Pa;            
    nk = nk + 1/alpha;    
    
    k=k+1;
end

B = B(:,1:k-1);
end