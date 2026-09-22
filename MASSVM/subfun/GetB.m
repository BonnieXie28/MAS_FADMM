function B = GetB(y,m,n)

    if  m>n
        s0 = ceil(n*(log(m/n))^2);
        T1 = find(y==1);  nT1= nnz(T1);
        T2 = find(y==-1); nT2= nnz(T2);

        if  nT1 < s0
            B = [T1; T2(1:(s0-nT1))];
        elseif nT2 < s0
            B = [T1(1:(s0-nT2)); T2];
        else
            B = [T1(1:ceil(s0/2)); T2(1:(s0-ceil(s0/2)))];
        end
        B = sort(B(1:s0));
    else
        B = 1:length(y);
    end
end
