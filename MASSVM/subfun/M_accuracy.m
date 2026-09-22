function [class,M_score_12] = M_accuracy(y, u_train, u_test, n_train, n_test, d)
[class_all, support] = assign_classes_mi_mii_miii(y, u_train, u_test, d);
class = class_all(:,1);
M_score_12 = squeeze(support(:,:,1));
end
