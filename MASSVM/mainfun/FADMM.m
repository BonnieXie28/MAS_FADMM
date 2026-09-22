function  out = FADMM(X,y,pars)

if nargin<3;               pars  = [];                             end

if isfield(pars,'maxit');  maxit = pars.maxit; else; maxit = 2000; end
if isfield(pars,'tol');    tol   = pars.tol;   else; tol   = 1e-3; end
if isfield(pars,'eta');    eta   = pars.eta;   else; eta   = 1;    end
if isfield(pars,'rho');    rho   = pars.rho;   else; rho   = 1;    end
if isfield(pars,'nu');     nu    = pars.nu;    else; nu    = 0.2;  end
if isfield(pars,'y_large_train'); y_large_train = pars.y_large_train; else; y_large_train = y; end
if isfield(pars,'record_history'); record_history = pars.record_history; else; record_history = false; end
if isfield(pars,'force_maxit'); force_maxit = logical(pars.force_maxit); else; force_maxit = false; end

[R,S] = size(X);
w     = ones(S,1)/100;
theta = zeros(R,1);
tau = 1;

theta_rho = theta;
eta_rho   = eta;

nu22  = nu^2/2;
B     = GetB(y,R,S);
size_B= numel(B);
B1    = B(1:ceil(size_B/2));
B2    = B(ceil(size_B/2)+1:size_B);
Q     = y.*X;

sigma = 1-Q*w-theta_rho;
vartheta = ones(1,3);
Fnorm    = @(var)norm(var)^2;
flag     = 0;

if record_history
    objective_history = nan(maxit,1);
    obj_gap_history = nan(maxit,1);
    relative_error_history = nan(maxit,1);
    kkt_residual_history = nan(maxit,1);
    working_set_size_history = nan(maxit,1);
else
    objective_history = [];
    obj_gap_history = [];
    relative_error_history = [];
    kkt_residual_history = [];
    working_set_size_history = [];
end
if record_history
    previous_objective = ccs_svm_objective(w,Q*w,eta,nu);
end
tACC = nan(maxit,1);

to       = tic;
for iter = 1:maxit
    w_old = w;


    tmp = 2*eta_rho/nu;
    if flag==1 || R<S || R>10000
        if eta_rho < nu22
            B1 = find(sigma >= 0 & sigma <= tmp);
            B2 = find(sigma > tmp & sigma < nu);
            B  = union(B1,B2);
        else
            B  = find(sigma >= 0 & sigma <= sqrt(2*eta_rho));
        end
    end

    if eta_rho <nu22
        mu = Prox1(sigma,eta_rho,nu,B1,B2);
    else
        mu = Prox2(sigma,eta_rho,B);
    end

    psi = 1 - mu - theta_rho;
    nB  = nnz(B);
    QB  = Q(B,:);
    if min(S,nB) < 1e3
        QBt = QB';
        if  S <= nB
            QQB = QBt*QB;
            QQB(1:1+S:end) = QQB(1:1+S:end) + 1/rho;
            w   = pinv(QQB)*(QBt*psi(B));


        else
            if nB > 0
                QQB = QB *QBt;
                QQB(1:1+nB:end) = QQB(1:1+nB:end) + 1/rho;
                w  = QBt*(QQB\psi(B));


            else
                w  = -ones(S,1); B=1;
            end
        end
    else
        if  S  <= nB
            w  = my_cg(QB,rho,(psi(B)'*QB)',S,1);
        else
            wB = my_cg(QB,rho,psi(B),nB,2);
            w  = (wB'*QB)';
        end
    end

    thetaB  = theta(B);
    theta   = zeros(R,1);
    Qw      = Q*w;
    phi     = mu + Qw - 1;
    thetaB  = thetaB + tau * rho * phi(B);
    theta(B)= thetaB;

    theta_rho = theta/rho;
    mu_theta  = mu  - theta_rho;
    if eta_rho <nu22
        prox_mu_theta = Prox1(mu_theta,eta_rho,nu,B1,B2);
    else
        prox_mu_theta = Prox2(mu_theta,eta_rho,B);
    end

    vartheta(1) = Fnorm(w'+ thetaB'*QB)/(1+Fnorm(w));
    vartheta(2) = Fnorm(phi)/(R+Fnorm(Qw));
    vartheta(3) = Fnorm(mu-prox_mu_theta)/(1+Fnorm(mu));
    stopping_error = max(vartheta);

    CPU         = toc(to);
    ACC         = 1-nnz(sign(y.*Qw)-y_large_train)/length(y);
    tACC(iter)  = ACC;
    if record_history
        current_objective = ccs_svm_objective(w,Qw,eta,nu);
        objective_history(iter) = current_objective;
        obj_gap_history(iter) = trajectory_objective_gap( ...
            current_objective,previous_objective);
        previous_objective = current_objective;
        relative_error_history(iter)     = trajectory_relative_error(w,w_old);
        kkt_residual_history(iter,:)    = stopping_error;
        working_set_size_history(iter)   = nB;
    end
    if ~force_maxit && stopping_error < tol && ACC>0.9; flag=2; break;  end

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
    eta_rho = eta/rho;

    if R<S || R>10000
        if rho > 2*eta/sig0^2
           rho = 1.9*eta/sig0^2;
        end
    end

    if iter>5 && R>S && R<=10000
        if std(tACC(iter-3:iter)) <= 1e-3
            m0 = max(20,2*S);
            B0 = find(sigma >= 0 & sigma < sqrt(2*eta_rho));
            if nnz(B0)<5 || nnz(B0)>=m0
                flag = 1;
                sigma1 = sigma(sigma>0);
                if nnz(sigma1) > m0
                    s   = mink(sigma1,m0);
                    rho = min(2*eta/s(m0)^2,10000);
                else
                    flag  = 0;
                    [~,B] = mink(abs(sigma),m0);
                end
            end
        end
    end
    eta_rho = eta/rho;
end

out.iter  = iter;
out.time  = CPU;
out.w     = w;
out.mu    = mu;
out.theta = theta;
out.nsv   = nB;
out.acc   = ACC;
out.error = stopping_error;
out.flag  = flag;
if record_history
    out.iteration_history = (1:iter)';
    out.objective_history = objective_history(1:iter);
    out.obj_gap_history = obj_gap_history(1:iter);
    out.relative_error_history = relative_error_history(1:iter);
    out.kkt_residual_history = kkt_residual_history(1:iter,:);
    out.working_set_size_history = working_set_size_history(1:iter);
else
    out.iteration_history = [];
    out.objective_history = [];
    out.obj_gap_history = [];
    out.relative_error_history = [];
    out.kkt_residual_history = [];
    out.working_set_size_history = [];
end
end
