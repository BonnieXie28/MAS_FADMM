clear; clc; close all;

project_root = fileparts(fileparts(mfilename('fullpath')));
Data = csvread(fullfile(project_root,'dataset','machine_ord.csv'), 1);

gamma_vec = [2 3 4];
leng = length(gamma_vec);

result = zeros(leng*2,13);

for ii = 1:leng
gamma_j = gamma_vec(ii);

m = size(Data,2)-1;
d = numel(unique(Data(:,1)));

X = Data(:,2:end);
y = Data(:,1);
p = 1;
[Xsample, ysample] = stratified_sampling(X, y, p, d);
Xnorm = normalize_matrix(Xsample);

n = size(Xnorm,1);
alpha   = zeros(m,1); beta = ones(m,1);
gamma = gamma_j*ones(m,1);

k = 5;
indices = crossvalind('Kfold', n, k);

fprintf(' ------------------------------------------------------------------------\n');
fprintf('      eta      rho     Iter      ACC       tACC       NSV        TIME    \n');
fprintf(' ------------------------------------------------------------------------\n');

mACC = zeros(9,9);
for i = -4:1:4
    pars.eta = 2^i;
    for j = -4:1:4
        pars.rho = 2^j;
        cv_acc   = zeros(k,1);
        for fold = 1:k

            test_idx  = (indices == fold);
            train_idx = ~test_idx;

            X_train = Xnorm(train_idx,:);
            y_train = ysample(train_idx);
            X_test  = Xnorm(test_idx,:);
            y_test  = ysample(test_idx);

            V_train = compute_V(X_train, alpha, beta, gamma);
            V_test  = compute_V(X_test, alpha, beta, gamma);
            [n_train,S] = size(V_train);
            n_test      = size(V_test,1);

            classes = cell(d,1);
            for h = 1:d
                classes{h} = V_train(y_train==h, :);
            end

            X_large = zeros(10000,S);r = 1;
            for h = d:-1:2
                V_h = classes{h};
                nh  = size(V_h,1);
                class_sub = h-1;
                for l = 1:class_sub
                    V_l = classes{l};
                    nl  = size(V_l,1);
                    for i_nh = 1:nh
                        for j_nl = 1:nl
                            X_large(r,:) = V_h(i_nh,:) - V_l(j_nl,:);
                            r = r + 1;
                        end
                    end
                end
            end
            X_plus  = X_large(1:r-1,:);
            y_plus  = ones(size(X_plus,1),1);
            X_minus = -X_plus;
            y_minus = -y_plus;
            X_large_train = [X_plus;X_minus];
            y_large_train = [y_plus;y_minus];

            classes = cell(d,1);
            for h = 1:d
                classes{h} = V_test(y_test==h, :);
            end

            X_large = zeros(10000,S);r = 1;
            for h = d:-1:2
                V_h = classes{h};
                nh  = size(V_h,1);
                class_sub = h-1;
                for l = 1:class_sub
                    V_l = classes{l};
                    nl  = size(V_l,1);
                    for i_nh = 1:nh
                        for j_nl = 1:nl
                            X_large(r,:) = V_h(i_nh,:) - V_l(j_nl,:);
                            r = r + 1;
                        end
                    end
                end
            end
            X_plus  = X_large(1:r-1,:);
            y_plus  = ones(size(X_plus,1),1);
            X_minus = -X_plus;
            y_minus = -y_plus;
            X_large_test = [X_plus;X_minus];
            y_large_test = [y_plus;y_minus];

            out = FADMM(X_large_train,y_large_train,pars);
            w   = out.w;
            u_test  = V_test*w;
            u_train = V_train*w;

            if out.flag == 2
                ACC_test = 1-nnz(sign(X_large_test*w)-y_large_test)/length(y_large_test);
                fprintf(' | %5.2f  |  %5.2f  |  %3d  |  %6.4f  |  %6.4f  |  %4d  |  %5.3fsec  |\n',...
                pars.eta, pars.rho, out.iter, out.acc, ACC_test, out.nsv, out.time);

                cv_acc(fold) = ACC_test;
            else
                cv_acc(fold) = 0;
            end
        end
        avg_acc       = mean(cv_acc);
        mACC(i+5,j+5) = avg_acc;
    end
end
[row, col] = find(mACC == max(max(mACC)));
i_best = row(1); j_best = col(1);
best_eta = 2^(i_best-5); best_rho = 2^(j_best-5);


rn = 50;
metrics_test = zeros(rn,3,4);
runtime_test = zeros(rn,1);
for repeat = 1:rn

mc = ceil(0.7*n);  mt = n-mc;        I  = randperm(n);
Tt = I(1:mt);      Xt = Xnorm(Tt,:); yt = ysample(Tt);
T  = I(mt+1:end);  X  = Xnorm(T,:);  y  = ysample(T,:);

V_train = compute_V(X, alpha, beta, gamma);
V_test  = compute_V(Xt, alpha, beta, gamma);
[n_train,S] = size(V_train);
n_test = size(V_test,1);

classes = cell(d,1);
for h = 1:d
    classes{h} = V_train(y==h, :);
end

X_large = zeros(10000,S);r = 1;
for h = d:-1:2
    V_h = classes{h};
    nh  = size(V_h,1);
    class_sub = h-1;
    for l = 1:class_sub
        V_l = classes{l};
        nl  = size(V_l,1);
        for i_nh = 1:nh
            for j_nl = 1:nl
                X_large(r,:) = V_h(i_nh,:) - V_l(j_nl,:);
                r = r + 1;
            end
        end
    end
end
X_plus  = X_large(1:r-1,:);
y_plus  = ones(size(X_plus,1),1);
X_minus = -X_plus;
y_minus = -y_plus;
X_large_train = [X_plus;X_minus];
y_large_train = [y_plus;y_minus];

classes = cell(d,1);
for h = 1:d
    classes{h} = V_test(yt==h, :);
end

X_large = zeros(10000,S);r = 1;
for h = d:-1:2
    V_h = classes{h};
    nh  = size(V_h,1);
    class_sub = h-1;
    for l = 1:class_sub
        V_l = classes{l};
        nl  = size(V_l,1);
        for i_nh = 1:nh
            for j_nl = 1:nl
                X_large(r,:) = V_h(i_nh,:) - V_l(j_nl,:);
                r = r + 1;
            end
        end
    end
end
X_plus  = X_large(1:r-1,:);
y_plus  = ones(size(X_plus,1),1);
X_minus = -X_plus;
y_minus = -y_plus;
X_large_test = [X_plus;X_minus];
y_large_test = [y_plus;y_minus];

pars.eta  = best_eta;
pars.rho  = best_rho;
final_out = FADMM(X_large_train, y_large_train, pars);
w         = final_out.w;
u_test    = V_test*w;
u_train   = V_train*w;

[predicted_class,~] = assign_classes_mi_mii_miii(y, u_train, u_test, d);
metrics_test(repeat,:,:) = multiclass_macro_metrics(yt, predicted_class, d);
runtime_test(repeat) = final_out.time;
end

metrics_mean = squeeze(mean(metrics_test,1));
metrics_std  = squeeze(std(metrics_test,0,1));
fprintf(' | %5.2f  |  %5.2f  |  %3d  |  %4d  |  %5.3fsec  |  %3d  |\n',...
   pars.eta, pars.rho, final_out.iter, final_out.nsv, final_out.time,final_out.flag);

result(ii*2-1,:) = [reshape(metrics_mean,1,[]), mean(runtime_test)];
result(ii*2,:) = [reshape(metrics_std,1,[]), std(runtime_test)];
end
