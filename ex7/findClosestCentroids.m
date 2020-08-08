function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K and m
K = size(centroids, 1);
M = size(X,1);
N = size(X,2);

% You need to return the following variables correctly.
idx = zeros(M, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

for i = 1:M
    dist_opt = 999;
    centroid_opt = 0;
    
    for k = 1:K
        dist = 0;
        
        for j = 1:N
            dist = dist + (X(i,j) - centroids(k,j)) ^ 2;
        end
        
        if dist < dist_opt
            dist_opt = dist;
            centroid_opt = k;
        end
    end
    idx(i) = centroid_opt;
end

% =============================================================

end

