function [X,y,w] = generated_samples(n,m,d,gamma_j,gamma,alpha, beta, flag)

S   = sum(gamma);
m_2 = ceil(m/2);
y = zeros(n,1);
X = rand(n,m);
V = compute_V(X, alpha, beta, gamma);

switch flag
    case 1
        w = rand(S,1);

    case 2
        w = rand(S,1);

        select_criteria = 1:m_2;
        for i = 1:m_2
            ms = select_criteria(i);
            w((ms-1)*gamma_j+1:ms*gamma_j) = - w((ms-1)*gamma_j+1:ms*gamma_j);
        end

    case 3
        w = randn(S,1);

    otherwise
        w = rand(S,1);

        select_criteria = 1:m_2;
        for i = 1:m_2
            ms = select_criteria(i);
            w((ms-1)*gamma_j+1:ms*gamma_j) = randn(gamma_j,1);
        end
end
U = V*w;


[~, id] = sort(U, 'ascend');
mod   = n/d;
class = cell(1, d);
for h = 1:d
    start_idx = (h-1)*mod + 1;
    end_idx   = min(h*mod, n);
    class{h}  = id(start_idx:end_idx);
end
for k = 1:d
    id1   = class{k};
    for i = 1:mod
    y(id1(i)) = k;
    end
end

end
