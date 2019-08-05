function out = mapFeature(X1, X2)
% MAP_FEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2) maps the two input features
%   to quadratic features used in the regularization exercise.
%
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%
%   Inputs X1, X2 must be the same size
%
degree = 6;
out = ones(size(X1(:,1)));  % it forms a vector "out" with the first column of 1
                            % because the first feature is always just 1
for i = 1:degree
    for j = 0:i
        % it adds to the matrix with ones on the first column, all the other 
        % columns with the value corresponding to the feature (possible 
        % combination of X1,X2 until the sixth degree)
       out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end
end

end