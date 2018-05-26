function B = ChangePitchScale(beta,A,Fs)

n_marks = size(A,2);
nk = 1;
ts = 1;

B = zeros(2,n_marks);

k=1;

while nk<=n_marks
    
    if A(2,floor(nk))==1
        scale = 1/beta;
    else
        scale = 1;
    end
    
    B(1,k) = floor(ts);
    B(2,k) = floor(nk);    

    ts = ts + scale*A(3,floor(nk));
    nk = nk + scale;    
    
    k = k+1;
end

B = B(:,1:k-1);
end