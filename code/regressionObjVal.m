function [error, error_grad] = regressionObjVal(w, X, y, lambda)

% compute squared error (scalar) and gradient of squared error with respect
% to w (vector) for the given data X and y and the regularization parameter
% lambda

N = size(X,1);
error_grad = 2*((transpose(X)*X)*w - (transpose(X)*y))/N;
error_grad = error_grad + 2.*lambda.*w;

error = (y - X*w);
error = error.^2;
error = sum(error,1);
error = error/N;
error = error + lambda*transpose(w)*w;

 