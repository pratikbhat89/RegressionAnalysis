function w = learnOLERegression(X,y)

% Implement OLE training here
% Inputs:
% X = N x D
% y = N x 1
% Output:
% w = D x 1

w_Transpose = transpose(X)*X;
w_Inverse = inv(w_Transpose);
w = w_Inverse*transpose(X)*y;