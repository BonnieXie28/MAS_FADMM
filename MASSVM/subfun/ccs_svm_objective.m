function value = ccs_svm_objective(w,Qw,eta,nu)
margin_residual = 1-Qw;
loss_value = zeros(size(margin_residual));
loss_value(margin_residual>=nu) = 1;
middle = margin_residual>0 & margin_residual<nu;
scaled = margin_residual(middle)/nu;
loss_value(middle) = 2*scaled-scaled.^2;
value = 0.5*(w'*w)+eta*sum(loss_value);
end
