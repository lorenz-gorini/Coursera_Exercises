function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n=length(theta);
% You need to return the following variables correctly 
J = 0;
grad = zeros(n,1); %column vector
% m=118 training samples
% n=28  features

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% This temp variable is useful because the grad(1)/theta(0) cannot be regulariz.
temp=theta;
temp(1)=0;
% Second term calculation used in the regularization
% y=column vector
sum=0;

sum=temp'*temp;

% Same calculation like the case without the regularization
exponent=X*theta; %column vector (m values)
h_x=sigmoid(exponent)'; %row vector (thanks to transpose)
%Cost function
J=-1/m*( y'*(log(h_x)') + (1-y)'*(log(1-h_x)') )+lambda/(2*m)*sum;

%----------GRADIENTS-----------
grad=(1/m)*((h_x'-y)'*X)'+(lambda/m)*temp;

% =============================================================
end
