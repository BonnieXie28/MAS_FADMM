function mu = Prox2_nB(sigma,eta_rho)
    mu = sigma;
    B = find(sigma >=0 & sigma < sqrt(2*eta_rho));
    mu(B) = 0;
end
