function NX = Normalization( X, normal_type)

if normal_type==0
    NX = X;

elseif normal_type==1
    C    = bsxfun(@minus, X, mean(X,2));
    Yrow = bsxfun(@rdivide, C, std(X,0,2));
    Y    = Yrow';
    D    = bsxfun(@minus, Y, mean(Y,2));
    Ycol = bsxfun(@rdivide, D, std(Y,0,2));
    NX   = Ycol';

else
    if normal_type==2
    nX = 1./max(abs(X),[],1);
    else
    nX = 1./sqrt(sum(X.*X));
    end

    lX = length(nX);
    if lX <= 10000
        NX  = X*sparse(1:lX,1:lX, nX,lX,lX);
    else
        k  = 5e3;
        if nnz(X)/lX/lX<1e-4; k = 1e5; end
        K      = ceil(lX/k);
        for i  = 1:K-1
        T      = ((i-1)*k+1):(i*k);
        X(:,T) =  X(:,T)*sparse(1:k,1:k,nX(T),k,k);
        end
        T      = ((K-1)*k+1):lX;
        k0     = length(T);

        X(:,T) = sparse(X(:,T))*sparse(1:k0,1:k0,nX(T),k0,k0);
        NX     = X;
    end
end
end
