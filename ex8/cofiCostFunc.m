function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it 
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% Cost:
sumSqrError = 0;
for i = 1:num_movies
    for j = 1:num_users
        if R(i,j) == 1
            sqrError = (X(i,:)*transpose(Theta(j,:)) - Y(i,j)) ^ 2;
            sumSqrError = sumSqrError + sqrError;
        end
    end
end
J = (1/2) * sumSqrError;

% 1. Regularizer for X:
reg_X = 0;
for i = 1:num_movies
    for k = 1:num_features
        reg_X = reg_X + X(i,k)^2;
    end
end
reg_X = (lambda/2) * reg_X;

% 2. Regularizer for Theta:
reg_Theta = 0;
for j = 1:num_users
    for k = 1:num_features
        reg_Theta = reg_Theta + Theta(j,k) ^ 2;
    end
end
reg_Theta = (lambda/2) * reg_Theta;

% Cost with Regularization:
J = J + reg_X + reg_Theta;



% Gradient with respect to X:
for i = 1:num_movies
    for j = 1:num_users
        if R(i,j) == 1
            X_grad(i,:) = X_grad(i,:) + ...
                (X(i,:)*transpose(Theta(j,:))-Y(i,j)) * Theta(j,:);
        end
    end
    X_grad(i,:) = X_grad(i,:) + lambda * X(i,:);
end


% Gradient with respect to Theta:
for j = 1:num_users
    for i = 1:num_movies
        if R(i,j) == 1
            Theta_grad(j,:) = Theta_grad(j,:) + ...
                (X(i,:)*transpose(Theta(j,:))-Y(i,j)) * X(i,:);
        end
    end
    Theta_grad(j,:) = Theta_grad(j,:) + lambda * Theta(j,:);
end


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
