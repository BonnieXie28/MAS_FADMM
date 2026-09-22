function x = my_cg(AT,sigma,b,n,flag)
    x = zeros(n,1);
    r = b;
    e = sum(r.*r);
    t = e;
    for i = 1: 50
        if e < 1e-10*t; break; end

        if  i == 1
            p = r;
        else
            p = r + (e/e0)*p;
        end

        if  flag==1
            w  = p/sigma  + ((AT*p)'*AT)';
        else
            w  = p/sigma  + AT*(p'*AT)';
        end

        a  = e/sum(p.*w);
        x  = x + a * p;
        r  = r - a * w;
        e0 = e;
        e  = sum(r.*r);
    end
end
