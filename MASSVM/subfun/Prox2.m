function mu = Prox2(sigma,eta_rho,B)
    mu = sigma;
    if isempty(B)
        B = find(sigma >=0 & sigma < sqrt(2*eta_rho));
        mu(B) = 0;
    else
        mu(B) = 0;
    end
end
