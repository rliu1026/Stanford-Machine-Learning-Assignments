function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Cost:
for i=1:m
    costPos = - y(i,:) * log(sigmoid(X(i,:) * theta));
    costNeg = - (1-y(i,:)) * log(1 - sigmoid(X(i,:) * theta));
    J = J + costPos + costNeg;
end
J = J ./ m ;

% Regularizer term in cost, but not regularizing theta(1):
reg = 0;
for j=2:size(theta)
    reg = reg + theta(j,1).^2;
end
reg = (lambda/(2*m)) * reg;
J = J + reg;

% Gradient: 
for j=1:size(theta)
    sum = 0;
    for k=1:m
        sum = sum + (sigmoid(X(k,:) * theta) - y(k,:)) .* X(k,j);
    end
    
    if j==1
        % Do not regularize theta(1) as it is always 1
        sum = (1/m) .* sum;
    else
        sum = (1/m) .* sum + lambda/m*theta(j,:);
    end
        
    grad(j,:) = sum;
end



% =============================================================

end
