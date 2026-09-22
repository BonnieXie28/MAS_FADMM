function [X_pair,y_pair] = build_pairwise_ordinal_data(V,y,d)

y = y(:);
if size(V,1) ~= numel(y)
    error('build_pairwise_ordinal_data:DimensionMismatch', ...
        'V and y must have the same number of rows.');
end
if any(~ismember(y,1:d))
    error('build_pairwise_ordinal_data:InvalidLabels', ...
        'y must contain ordinal labels in 1:d.');
end

class_size = accumarray(y,1,[d,1]);
n_positive = 0;
for high = 2:d
    for low = 1:high-1
        n_positive = n_positive + class_size(high)*class_size(low);
    end
end

X_positive = zeros(n_positive,size(V,2));
cursor = 1;
for high = 2:d
    V_high = V(y==high,:);
    for low = 1:high-1
        V_low = V(y==low,:);
        block_size = size(V_high,1)*size(V_low,1);
        rows = cursor:cursor+block_size-1;
        X_positive(rows,:) = repelem(V_high,size(V_low,1),1) - ...
            repmat(V_low,size(V_high,1),1);
        cursor = cursor+block_size;
    end
end

X_pair = [X_positive;-X_positive];
y_pair = [ones(n_positive,1);-ones(n_positive,1)];
end
