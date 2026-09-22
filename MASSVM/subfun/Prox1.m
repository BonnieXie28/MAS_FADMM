function mu = Prox1(sigma,eta_rho,nu,B1,B2)
    tmp = 2*eta_rho/nu;
    mu  = sigma;
    B   = union(B1,B2);
    if isempty(B)
        B1     = find(sigma >= 0 & sigma <= tmp);
        mu(B1) = 0;
        B2     = find(sigma > tmp & sigma < nu);
        sigma2 = sigma(B2);
        mu(B2) = (sigma2 - tmp)/(1 - tmp/nu);
    else
        mu(B1) = 0;
        sigma2 = sigma(B2);
        mu(B2) = (sigma2 - tmp)/(1 - tmp/nu);
    end
end
