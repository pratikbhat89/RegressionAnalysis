% load the data
load diabetes;
x_train_i = [ones(size(x_train,1),1) x_train];
x_test_i = [ones(size(x_test,1),1) x_test];

%%% FILL CODE FOR PROBLEM 1 %%%
% linear regression without intercept
w_no_intercept = learnOLERegression(x_train,y_train);

error_linear_train = y_train - x_train*w_no_intercept;
error_linear_train = error_linear_train.^2;
error_linear_train = sum(error_linear_train,1);
error_linear_train = error_linear_train^0.5;

error_linear_test = y_test - x_test*w_no_intercept;
error_linear_test = error_linear_test.^2;
error_linear_test = sum(error_linear_test,1);
error_linear_test = error_linear_test^0.5;


% linear regression with intercept
w_intercept = learnOLERegression(x_train_i,y_train);

error_linear_intercept_train = y_train - x_train_i*w_intercept;
error_linear_intercept_train = error_linear_intercept_train.^2;
error_linear_intercept_train = sum(error_linear_intercept_train,1);
error_linear_intercept_train = error_linear_intercept_train^0.5;

error_linear_intercept_test = y_test - x_test_i*w_intercept;
error_linear_intercept_test = error_linear_intercept_test.^2;
error_linear_intercept_test = sum(error_linear_intercept_test,1);
error_linear_intercept_test = error_linear_intercept_test^0.5;

%%% END PROBLEM 1 CODE %%%



%%% FILL CODE FOR PROBLEM 2 %%%
% ridge regression using least squares - minimization

lambdas = 0:0.00001:0.001;

train_errors_ridge = zeros(length(lambdas),1);
test_errors_ridge = zeros(length(lambdas),1);

for i = 1:length(lambdas)
    lambda = lambdas(i);
    
    % fill code here for prediction and computing errors
    w = learnRidgeRegression(x_train_i,y_train,lambda);
    
    error_ridge_train = (y_train - x_train_i*w);
    error_ridge_train = error_ridge_train.^2;
    error_ridge_train = sum(error_ridge_train,1);
    error_ridge_train = (error_ridge_train)^0.5;
    train_errors_ridge(i,1) = error_ridge_train;
    
    error_ridge_test = (y_test - x_test_i*w);
    error_ridge_test = error_ridge_test.^2;
    error_ridge_test = sum(error_ridge_test,1);
    error_ridge_test = (error_ridge_test)^0.5;
    test_errors_ridge(i,1) = error_ridge_test;
end
figure;
plot([train_errors_ridge test_errors_ridge]);
legend('Training Error','Testing Error');

%Fnding the optimal lambda value for problem 4.
[test_error_min_ridge test_error_min_index_ridge] = min(test_errors_ridge(:));
lamda_optimal_ridge = lambdas(1,test_error_min_index_ridge);

%Finding the weight vector for optimal lambda for compariosn with weight
%vector in problem 1
w_ridge_lambda_optimal = learnRidgeRegression(x_train_i,y_train,lamda_optimal_ridge);

%%% END PROBLEM 2 CODE %%%

%%% BEGIN PROBLEM 3 CODE
% ridge regression using gradient descent - see handouts (lecture 21 p5) or
% http://cs229.stanford.edu/notes/cs229-notes1.pdf (page 11)
initialWeights = zeros(65,1);
% set the maximum number of iteration in conjugate gradient descent
options = optimset('MaxIter', 500);

% define the objective function
lambdas = 0:0.00001:0.001;
train_errors_gradient = zeros(length(lambdas),1);
test_errors_gradient = zeros(length(lambdas),1);

% run ridge regression training with fmincg
for i = 1:length(lambdas)
    lambda = lambdas(i);
    objFunction = @(params) regressionObjVal(params, x_train_i, y_train, lambda);
    w = fmincg(objFunction, initialWeights, options);
    % fill code here for prediction and computing errors
    
    error_gradient_train = (y_train - x_train_i*w);
    error_gradient_train = error_gradient_train.^2;
    error_gradient_train = sum(error_gradient_train,1);
    error_gradient_train = (error_gradient_train)^0.5;
    train_errors_gradient(i,1) = error_gradient_train;
    
    error_gradient_test = (y_test - x_test_i*w);
    error_gradient_test = error_gradient_test.^2;
    error_gradient_test = sum(error_gradient_test,1);
    error_gradient_test = (error_gradient_test)^0.5;
    test_errors_gradient(i,1) = error_gradient_test;
end
figure;
plot([train_errors_gradient test_errors_gradient]);
legend('Training Error','Testing Error');

%Fnding the optimal lambda value for gradient descent.
[test_error_min_gradient test_error_min_index_gradient] = min(test_errors_gradient(:));
lamda_optimal_gradient = lambdas(1,test_error_min_index_gradient);

%%% END PROBLEM 3 CODE

%%% BEGIN  PROBLEM 4 CODE
% using variable number 3 only
x_train = x_train(:,3);
x_test = x_test(:,3);

train_errors_nonlinear = zeros(length(7),1);
test_errors_nonlinear = zeros(length(7),1);

train_errors_regularization_nonlinear = zeros(length(7),1);
test_errors_regularization_nonlinear = zeros(length(7),1);

% no regularization
lambda = 0;
for d = 0:6
    x_train_n = mapNonLinear(x_train,d);
    x_test_n = mapNonLinear(x_test,d);
    w = learnRidgeRegression(x_train_n,y_train,lambda);
    % fill code here for prediction and computing errors
    
    error_map_train = (y_train - x_train_n*w);
    error_map_train = error_map_train.^2;
    error_map_train = sum(error_map_train,1);
    error_map_train = (error_map_train)^0.5;
    train_errors_nonlinear(d+1,1) = error_map_train;
    
    error_map_test = (y_test - x_test_n*w);
    error_map_test = error_map_test.^2;
    error_map_test = sum(error_map_test,1);
    error_map_test = (error_map_test)^0.5;
    test_errors_nonlinear(d+1,1) = error_map_test;
end
figure;
plot([train_errors_nonlinear test_errors_nonlinear]);
legend('Training Error','Testing Error');

% optimal regularization. Lambda optimal found in problem 2. 
lambda = lamda_optimal_ridge;
for d = 0:6
    x_train_n = mapNonLinear(x_train,d);
    x_test_n = mapNonLinear(x_test,d);
    w = learnRidgeRegression(x_train_n,y_train,lambda);
    % fill code here for prediction and computing errors
    
    error_map_train = (y_train - x_train_n*w);
    error_map_train = error_map_train.^2;
    error_map_train = sum(error_map_train,1);
    error_map_train = (error_map_train)^0.5;
    train_errors_regularization_nonlinear(d+1,1) = error_map_train;
    
    error_map_test = (y_test - x_test_n*w);
    error_map_test = error_map_test.^2;
    error_map_test = sum(error_map_test,1);
    error_map_test = (error_map_test)^0.5;
    test_errors_regularization_nonlinear(d+1,1) = error_map_test;
end
figure;
plot([train_errors_regularization_nonlinear test_errors_regularization_nonlinear]);
legend('Training Error','Testing Error');

