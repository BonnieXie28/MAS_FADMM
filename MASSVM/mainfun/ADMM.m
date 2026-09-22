function  out = ADMM(X,y,pars)

if nargin<3;               pars  = [];                             end

if isfield(pars,'maxit');  maxit = pars.maxit; else; maxit = 2000; end
if isfield(pars,'tol');    tol   = pars.tol;   else; tol   = 1e-3; end
if isfield(pars,'eta');    eta   = pars.eta;   else; eta   = 1;    end
if isfield(pars,'rho');    rho   = pars.rho;   else; rho   = 1;    end
if isfield(pars,'nu');     nu    = pars.nu;    else; nu    = 0.2;  end
if isfield(pars,'y_large_train'); y_large_train = pars.y_large_train; else; y_large_train = y; end
if isfield(pars,'record_history'); record_history = logical(pars.record_history); else; record_history = false; end
if isfield(pars,'force_maxit'); force_maxit = logical(pars.force_maxit); else; force_maxit = false; end

[R,S] = size(X);
w     = ones(S,1)/100;
theta = zeros(R,1);

theta_rho = theta;
eta_rho   = eta;

nu22  = nu^2/2;
Q     = y.*X;
Qt    = Q';
sigma = 1-Q*w-theta_rho;
QtQ   = Qt*Q;
I     = eye(S);

vartheta = ones(1,3);

Fnorm    = @(var)norm(var)^2;
flag     = 0;

if record_history
    objective_history = nan(maxit,1);
    obj_gap_history = nan(maxit,1);
    relative_error_history = nan(maxit,1);
    kkt_residual_history = nan(maxit,1);
else
    objective_history = [];
    obj_gap_history = [];
    relative_error_history = [];
    kkt_residual_history = [];
end
if record_history
    previous_objective = ccs_svm_objective(w,Q*w,eta,nu);
end
tACC = nan(maxit,1);
to       = tic;
for iter = 1:maxit
    w_old = w;
    if eta_rho <nu22
        mu = Prox1_nB(sigma,eta_rho,nu);
    else
        mu = Prox2_nB(sigma,eta_rho);
    end

    psi = 1 - mu - theta_rho;
    w   = rho*pinv(I+rho*QtQ)*(Qt*psi);

    Qw    = Q*w;
    phi   = mu + Qw - 1;
    theta = theta + rho * phi;

    theta_rho = theta/rho;
    mu_theta  = mu  - theta_rho;
    if eta_rho <nu22
        prox_mu_theta = Prox1_nB(mu_theta,eta_rho,nu);
    else
        prox_mu_theta = Prox2_nB(mu_theta,eta_rho);
    end

    vartheta(1) = Fnorm(w'+ theta'*Q)/(1+Fnorm(w));
    vartheta(2) = Fnorm(phi)/(R+Fnorm(Qw));
    vartheta(3) = Fnorm(mu-prox_mu_theta)/(1+Fnorm(mu));
    error       = max(vartheta);

    CPU         = toc(to);
    ACC         = 1-nnz(sign(y.*Qw)-y_large_train)/length(y);
    tACC(iter)  = ACC;
    if record_history
        current_objective = ccs_svm_objective(w,Qw,eta,nu);
        objective_history(iter) = current_objective;
        obj_gap_history(iter) = trajectory_objective_gap( ...
            current_objective,previous_objective);
        previous_objective = current_objective;
        relative_error_history(iter) = trajectory_relative_error(w,w_old);
        kkt_residual_history(iter,:) = error;
    end
    if ~force_maxit && error < tol && ACC>=0.9 ; flag=2; break; end

    sigma = mu_theta - phi;

    sig0  = min(sigma(sigma>0));
    if isempty(sig0); sig0 = 1e-8; end

    if mod(iter,10)==0
        if vartheta(3) > 1e-3
           rho = min(rho*2,10);
        elseif vartheta(1) > 1e-3
           rho = min(max(0.01,rho/1.25),5);
        end
    end

    if R<S || R>10000
        if rho > 2*eta/sig0^2
           rho = 1.9*eta/sig0^2;
        end
    end
    eta_rho = eta/rho;
end

out.iter  = iter;
out.time  = CPU;
out.w     = w;
out.mu    = mu;
out.theta = theta;
out.acc   = ACC;
out.error = error;
out.flag  = flag;
if record_history
    out.iteration_history = (1:iter)';
    out.objective_history = objective_history(1:iter);
    out.obj_gap_history = obj_gap_history(1:iter);
    out.relative_error_history = relative_error_history(1:iter);
    out.kkt_residual_history = kkt_residual_history(1:iter,:);
else
    out.iteration_history = [];
    out.objective_history = [];
    out.obj_gap_history = [];
    out.relative_error_history = [];
    out.kkt_residual_history = [];
end
end
