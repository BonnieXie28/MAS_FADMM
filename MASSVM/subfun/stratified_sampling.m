function [new_X, new_y] = stratified_sampling(X, y, p, d)
    new_X = [];
    new_y = [];

    for i = 1:d
        idx = find(y == i);
        n   = length(idx);
        k   = ceil(p * n);

        sampled_idx = idx(randperm(n, k));
        new_X = [new_X; X(sampled_idx, :)];
        new_y = [new_y; y(sampled_idx)];
    end

    shuffled_idx = randperm(size(new_X, 1));
    new_X = new_X(shuffled_idx, :);
    new_y = new_y(shuffled_idx, :);
end
