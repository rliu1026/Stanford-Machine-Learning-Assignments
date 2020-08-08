function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n = size(X, 2); % number of features

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% Cost computation: 

sumError = 0;
sumThetaSqr = 0;

% Error:
for i = 1:m
    sumError = sumError + (X(i,:)*theta - y(i,:))^2;
end

% Regularizer:
for j = 2:n % Do not regularize theta(1), which is θ0.
    sumThetaSqr = sumThetaSqr + theta(j,:)^2;
end

J = (1/(2*m)) * sumError + (lambda/(2*m)) * sumThetaSqr;


% Gradient computation:

for j = 1:n
    
    sumGrad = 0;
    for i = 1:m
        sumGrad = sumGrad + (X(i,:)*theta - y(i,:)) * X(i,j);
    end
    
    grad(j,:) = (1/m) * sumGrad;
    if j ~= 1
        grad(j,:) = grad(j,:) + (lambda/m) * theta(j,:);
    end
    
end
        







% =========================================================================

grad = grad(:);

end
