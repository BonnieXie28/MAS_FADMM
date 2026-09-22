function [predicted_class, support] = assign_classes_mi_mii_miii(y_ref, u_ref, u_query, d)

y_ref = y_ref(:);
u_ref = u_ref(:);
u_query = u_query(:);

n_ref = numel(y_ref);
n_query = numel(u_query);

mi_ref = zeros(n_ref, 1);
for i = 1:n_ref
    h = y_ref(i);
    other = (1:n_ref)' ~= i;
    denom = sum(other & y_ref ~= h);
    if denom == 0
        continue;
    end
    confirms = other & ((y_ref < h & u_ref < u_ref(i)) | (y_ref > h & u_ref > u_ref(i)));
    mi_ref(i) = sum(confirms) / denom;
end

support = zeros(n_query, d, 3);
for q = 1:n_query
    uq = u_query(q);
    for h = 1:d
        outside = y_ref ~= h;
        denom = sum(outside);
        confirms = (y_ref < h & u_ref < uq) | (y_ref > h & u_ref > uq);

        if denom > 0
            support(q,h,1) = sum(confirms) / denom;

            support(q,h,2) = sum(mi_ref(confirms)) / denom;
        end

        members = find(y_ref == h);
        if isempty(members)
            continue;
        end
        direct_support = 0;
        for k = 1:numel(members)
            r = members(k);
            if uq >= u_ref(r)
                interval = u_ref >= u_ref(r) & u_ref <= uq;
            else
                interval = u_ref > uq & u_ref <= u_ref(r);
            end
            interval_size = sum(interval);
            if interval_size > 0
                direct_support = direct_support + sum(interval & y_ref == h) / interval_size;
            end
        end
        support(q,h,3) = direct_support / numel(members);
    end
end

[~, predicted_class] = max(support, [], 2);
predicted_class = squeeze(predicted_class);
if n_query == 1
    predicted_class = reshape(predicted_class, 1, 3);
end
end
