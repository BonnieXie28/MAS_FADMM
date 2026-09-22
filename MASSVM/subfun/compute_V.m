function V_matrix = compute_V(X, alpha, beta, gamma)
    [n, m] = size(X);
    S = sum(gamma);
    V_matrix = zeros(n,S);

    x_breaks = cell(m, 1);
    for j = 1:m
        x_breaks{j} = linspace(alpha(j), beta(j), gamma(j)+1);

    end

    for i = 1:n
        X_i = X(i, :);
        V  = [];

        for j = 1:m
            g_j = X_i(j);
            x_j = x_breaks{j};

            v_j = zeros(1, gamma(j));
            for t = 1:gamma(j)
                x_low  = x_j(t);
                x_high = x_j(t+1);

                if g_j >= x_high
                    v_j(t) = 1;
                elseif g_j >= x_low && g_j <= x_high
                    v_j(t) = (g_j - x_low) / (x_high - x_low);
                else
                    v_j(t) = 0;
                end
            end
            V = [V, v_j];
        end
        V_matrix(i,:) = V;
    end
end
