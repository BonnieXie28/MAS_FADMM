function metrics = multiclass_macro_metrics(y_true, y_pred, d)

y_true = y_true(:);
if isvector(y_pred)
    y_pred = y_pred(:);
end

n_rules = size(y_pred,2);
metrics = zeros(n_rules, 4);
for r = 1:n_rules
    pred = y_pred(:,r);
    metrics(r,1) = mean(pred == y_true);
    precision = zeros(d,1);
    recall = zeros(d,1);
    f1 = zeros(d,1);
    for h = 1:d
        tp = sum(y_true == h & pred == h);
        fp = sum(y_true ~= h & pred == h);
        fn = sum(y_true == h & pred ~= h);
        if tp + fp > 0
            precision(h) = tp / (tp + fp);
        end
        if tp + fn > 0
            recall(h) = tp / (tp + fn);
        end
        if precision(h) + recall(h) > 0
            f1(h) = 2 * precision(h) * recall(h) / (precision(h) + recall(h));
        end
    end
    metrics(r,2:4) = [mean(precision), mean(recall), mean(f1)];
end
end
