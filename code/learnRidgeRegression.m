function w = learnRidgeRegression(X,y,lambda)

% Implement ridge regression training here
% Inputs:
% X = N x D
% y = N x 1
% lambda = scalar
% Output:
% w = D x 1


w_Transpose = (size(X,1)*lambda*eye(size(X,2))) + (transpose(X)*X);
w_Inverse = inv(w_Transpose);
w = w_Inverse*transpose(X)*y;