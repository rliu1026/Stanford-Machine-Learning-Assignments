function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
X = [ones(m, 1) X];

for i = 1:size(X, 1)
    
    % Get hidden layer:
    % e.g. hidden(3) is the value of the 3rd node in the hidden layer
    hidden = zeros(size(Theta1, 1), 1);
    for j = 1:size(Theta1, 1)
        hidden(j) = sigmoid(Theta1(j, :) * transpose(X(i, :)));
    end
    
    hidden = [1; hidden]; % add bias unit
    
    % Get output layer:
    output = zeros(num_labels, 1);
    for k = 1:num_labels
        output(k) = sigmoid(Theta2(k, :) * hidden);
    end
    [val, p(i)] = max(output, [], 1); 
        % Find the index of the largest element
        % Each element is the probability of y being the index
end






% =========================================================================


end
