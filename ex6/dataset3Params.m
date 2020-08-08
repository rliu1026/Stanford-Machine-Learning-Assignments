function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_list = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_list = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
error_all = zeros(size(C_list, 1), size(sigma_list, 1));
error_opt = 1;

for p = 1:size(C_list, 1)
    for q = 1:size(sigma_list, 1)
        
        C_test = C_list(p);
        sigma_test = sigma_list(q);
        
        % Train the model:
        model = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));
        
        % Test the model on validation set:
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval));
        error_all(p,q) = error;
        
        % Compare parameters with previous ones:
        if error_opt > error
            error_opt = error;
            C = C_test;
            sigma = sigma_test;
        end
        
    end
end

% Visualize all errors:
%error_all

% =========================================================================

end
