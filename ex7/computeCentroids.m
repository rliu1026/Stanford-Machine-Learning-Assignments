function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[M N] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, N);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

num_members = zeros(K, 1);
   % num_members(k) = the number of examples assigned to cluster k.

% Sum up each feature of each example in each cluster. 
for i = 1:M
    k = idx(i);
    num_members(k) = num_members(k) + 1;
    
    for j = 1:N
        centroids(k,j) = centroids(k,j) + X(i,j);
    end
end
% For now, 
% centroids(k,j) = sum of feature j of all examples assigned to cluster k.

% Get average of each feature for every cluster:
for j = 1:N
    centroids(:,j) = centroids(:,j) ./ num_members;
end

% =============================================================


end

