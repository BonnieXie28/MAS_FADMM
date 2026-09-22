function mu = Prox1_nB(sigma,eta_rho,nu)
    tmp = 2*eta_rho/nu;
    mu  = sigma;

    B1     = find(sigma >= 0 & sigma <= tmp);
    mu(B1) = 0;

    B2     = find(sigma > tmp & sigma < nu);
    sigma2 = sigma(B2);
    mu(B2) = (sigma2 - tmp)/(1 - tmp/nu);
end
