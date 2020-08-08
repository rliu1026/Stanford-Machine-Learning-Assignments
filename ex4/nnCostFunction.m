function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

hypVec= zeros(m, num_labels);   
    % hypothesis prediction in 0s and 1s only, 
    % not involving in cost computation
hyp = zeros(m, num_labels); 
    % hypothesis of probabilities for each label
    % e.g. hyp(3,9) = predicted prob for example 3 to be label 9.
hidden = zeros(m, hidden_layer_size);   
    % values for hidden units
    % e.g. hidden(3,9) = value of the 9th hidden unit for example 3 
yVec = zeros(m, num_labels);
    % y in form of vectors 
    % e.g. y(3,:) = [0 0 0 0 0 0 0 0 0 1], example 3 belongs to label 10. 
cost = 0; % cost part of the cost function
reg = 0; % regularizer of the cost function

% Feedforward pass:
% 1. Compute values of hidden units:
X = [ones(m,1) X]; % Add bias units
for i = 1:m
    for j = 1:hidden_layer_size
        z = 0;
        for h = 1:input_layer_size + 1
            z = z + Theta1(j,h) * X(i,h);
        end
        hidden(i,j) = sigmoid(z);
    end
end

% 2. Compute values of output units:
hidden = [ones(m,1) hidden];
for i = 1:m
    for j = 1:num_labels
        z = 0;
        for h = 1:hidden_layer_size + 1
            z = z + Theta2(j,h) * hidden(i,h);
        end
        hyp(i,j) = sigmoid(z);
    end
end

% 3. Vectorize hyp into hypVec and y into yVec:
for i = 1:m 
    [val, maxIndex] = max(hyp(i,:), [], 1);
    hyp_case = zeros(1, num_labels);
    hyp_case(1, maxIndex) = 1;
    hypVec(i,:) = hyp_case;
    
    y_case = zeros(1, num_labels);
    y_case(1, y(i)) = 1;
    yVec(i,:) = y_case;
end

% Feedforward ends

% Cost computation: 
for i = 1:m
    for k = 1:num_labels
        cost = cost - yVec(i,k) * log(hyp(i,k)) - (1-yVec(i,k)) * log(1-(hyp(i,k)));
    end
end

J = (1/m) * cost; % Cost without regularizer

% Regularizer:
for h = 1:input_layer_size
    for j = 1:hidden_layer_size
        reg = reg + Theta1(j,h+1)^2; 
        % Using h+1 so that parameters connecting bias units are excluded
    end
end
for j = 1:hidden_layer_size
    for k = 1:num_labels
        reg = reg + Theta2(k,j+1)^2;
        % Using j+1 so that parameters connecting bias units are excluded
    end
end

X = X(:, 2:end);
J = (1/m) * cost + (lambda/(2*m)) * reg; % Cost with regularizer

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

X = [ones(m,1) X];
Triangle1 = zeros(size(Theta1));
Triangle2 = zeros(size(Theta2));

for t = 1:m
    
    output = zeros(num_labels, 1);
    deltaOutput = zeros(num_labels, 1);
    
    hidden = zeros(hidden_layer_size, 1);
    deltaHidden = zeros(hidden_layer_size, 1);
    z2Vec = zeros(size(hidden)); 
        % vector of hidden units' sum values before passing into sigmoid function
        
    % Feed forward pass:
    for j = 1:hidden_layer_size
        z = 0;
        for i = 1:input_layer_size + 1
            z = z + Theta1(j,i) * X(t,i);
        end
        hidden(j,1) = sigmoid(z);
        z2Vec(j,1) = z;
    end
    
    hidden = [1; hidden];
    z2Vec = [1; z2Vec];
    
    for k = 1:num_labels
        z = 0;
        for j = 1:hidden_layer_size + 1
            z = z + Theta2(k,j) * hidden(j,1);
        end
        output(k,1) = sigmoid(z);
    end
    
    % Compute delta:
    for k = 1:num_labels
        deltaOutput(k,1) = output(k,1) - yVec(t,k);
    end
    deltaHidden = (transpose(Theta2) * deltaOutput) .* sigmoidGradient(z2Vec);
    
    deltaHidden = deltaHidden(2:end);
    
    Triangle1 = Triangle1 + deltaHidden * X(t,:);
    Triangle2 = Triangle2 + deltaOutput * transpose(hidden);
    
end
    
% Theta1_grad = (1/m) * Triangle1;
% Theta2_grad = (1/m) * Triangle2;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

regTheta1 = Theta1;
regTheta2 = Theta2;
regTheta1(:,1) = 0;
regTheta2(:,1) = 0;
Theta1_grad = (1/m) * Triangle1 + (lambda/m) * regTheta1;
Theta2_grad = (1/m) * Triangle2 + (lambda/m) * regTheta2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
