function B = ChangeBothScales(alpha,beta,A,Fs)

n_marks = size(A,2);
ts = 1;
nk = 1;

B = zeros(2,n_marks);

k=1;

while nk<=n_marks
    
    Pa = A(3,floor(nk));
    
    B(1,k) = floor(ts);
    B(2,k) = floor(nk);
    
    if A(2,floor(nk))==1
        scale = 1/beta;
    else
        scale = 1;
    end
    
    ts = ts + scale*Pa;
    nk = nk + scale/alpha;   
    
    k=k+1;
end

B = B(:,1:k-1);

end