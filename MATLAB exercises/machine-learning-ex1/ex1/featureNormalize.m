function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = double(zeros(1, size(X, 2)));
sigma = double(zeros(1, size(X, 2)));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by its standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%    
feat_num=length(X(1,:));
for feat=1:feat_num
    mu(feat)=double(mean(X(:,feat)));
    sigma(feat)=double(std(X(:,feat)));
    example_num=length(X(:,feat));
    for example=1:example_num
        X_norm(example,feat)=double((X(example,feat)-mu(feat))/sigma(feat));
    endfor
endfor


% ============================================================

end
