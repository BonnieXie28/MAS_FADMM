function X_normalized = normalize_matrix(X)

    min_vals = min(X, [], 1);
    max_vals = max(X, [], 1);

    range = max_vals - min_vals;
    zero_range = (range==0);

    X_normalized = zeros(size(X));
    X_normalized(:,zero_range) = 0.5;
    if any(~zero_range)
        X_normalized(:,~zero_range) = (X(:,~zero_range) - min_vals(:,~zero_range))./range(:,~zero_range);
    end

    X_normalized = max(0,min(1,X_normalized));
end
