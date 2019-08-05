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
% Reshape nn_params back into the parameters Theta1 and Theta2 (they are one 
% after another), the weight matrices for our 2 layer neural network 
% (number of elements)

%ME: First it was just ONE SINGLE row with that size.  
% Now we reshaped into a matrix of those dimensions.
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

%These theta matrices are the weights corresponding to the first and
% the second hidden layer                
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);  % = 5000
% input_layer_size = 400
% size(X) = 5000 x 400
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
% ==============================
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% (A_3 actually describes the probability that the picture 
% represents that specific label for every label i.e. 
% (0.8573,0.1231,0.0302,...)--> the output is the "label 0"=(1,0,0,0,0,...)

% when I consider y, I consider a vector with 0/1 
%size(output) = 5000x1



%===============================
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should 
%         
%          return the partial derivatives of
%         
%          the cost function with respect to Theta1 and Theta2 in Theta1_grad and
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
%
%    I need to transform y in a matrix for all the labels as rows
%     This way we obtain a matrix with 0s and 1s (for the correct label)
%

% We add the "x0 = +1" at the beginning
X=[ones(m,1) X];
%-----------INITIALIZE A_2---------
% A_2 must be the matrix with the hidden nodes (columns) of the first layer for 
% all the samples (rows)
##A_2 = zeros(hidden_layer_size+1); % size(A_2)= 5000x25 poi diventa  5000x26 grazie a x0=1
##A_3 = zeros(m,num_labels); % size(A_3)= 5000*10 

y_matrix=zeros(m,num_labels); %initialize y matrix
##for i=1:num_labels
##  y_matrix(:,i)=(y==i);  % a column vector which contains: 
                  % 1 for all the samples with the "output == label c" , 
                  % 0 otherwise (they will be 1 in the other cycles)
##endfor
for i=1:m
  y_matrix(i,y(i))=1;
endfor

A_2 = zeros(m,hidden_layer_size+1); % size(A_2)= 5000x25 poi diventa  5000x26 grazie a x0=1
A_3 = zeros(m,num_labels); % size(A_3)= 5000*10 
z_2 = zeros(m,hidden_layer_size); % size(A_2)= 5000x25 poi diventa  5000x26 grazie a x0=1
z_3 = zeros(m,num_labels);
% Forward Propagation
for i=1:m
  z_2(i,:)=(Theta1*X(i,:)')'; % vector(25)
  A_2(i,:)=[1 sigmoid(z_2(i,:))];  # (25x(400+1)*(400+1))'= vector(25+1)
  z_3(i,:)=(Theta2*A_2(i,:)')';	
  A_3(i,:)=sigmoid(z_3(i,:)); # ((25+1)x10*(25+1))'= vector(10)=result
endfor
% Cost function
for c=1:num_labels
  output=(y==c);  
  J=J+(-1/m)*( output'*(log(A_3(:,c))) + (1-output)'*(log(1-A_3(:,c))) );
endfor

% Cost Regularization
% Note we should not regularize the terms that correspond to the bias. 
% For the matrices Theta1 and Theta2, this corresponds to the first column of each matrix.
t1 = Theta1(:,2:size(Theta1,2));
t2 = Theta2(:,2:size(Theta2,2));
T1_unrolled=[t1(:)];
T2_unrolled=[t2(:)];

J=J+lambda/(2*m)*(T1_unrolled'*T1_unrolled+T2_unrolled'*T2_unrolled);

A_1 = zeros(m,input_layer_size+1); % vector 400+1
A_2 = zeros(m,hidden_layer_size+1); % size(A_2)= 5000x25 poi diventa  5000x26 grazie a x0=1
A_3 = zeros(num_labels); % size(A_3)= 5000*10 

% initialize deltas
delta_2=zeros(m,hidden_layer_size+1);
delta_3=zeros(m,num_labels);

Delta_major_1=zeros(hidden_layer_size,input_layer_size+1);
Delta_major_2=zeros(num_labels,hidden_layer_size+1);

for i=1:m
  % Forward Propagation
  A_1=X(i,:); % vector (401)
  z_2=(Theta1*A_1')'; % (25x(400+1)*(400+1))'= vector(25)
  A_2=[1; sigmoid(z_2)']; % row vector(25+1)
  z_3=(Theta2*A_2)';  % ((25+1)x10*(25+1))'= vector(10)
  A_3=sigmoid(z_3); % vector(10)=result

  % Back Propagation
  delta_3=A_3-y_matrix(i,:); # vector 1x10
  z_2=[1; z_2'];
  delta_2=(delta_3*Theta2)'.*sigmoidGradient(z_2); #(25+1)x10 * 1x10 = vector 1x26
  delta_2=delta_2(2:end);
  Delta_major_1=Delta_major_1+delta_2*A_1; #matrix 25x401 skipping z2(0)=(25*1)
  Delta_major_2=Delta_major_2+delta_3'*A_2'; #matrix 10x26
endfor

% Regularized gradients
% Step 5
Theta2_grad = (1/m) * Delta_major_2; % (10*26)
Theta1_grad = (1/m) * Delta_major_1; % (25*401)

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end)); % for j >= 1 
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end)); % for j >= 1

##Theta1_grad=1/m*(Delta_major_1);  %J = 0
##Theta1_grad(:,2:end)=Theta1_grad(:,2:end)+1/m*(lambda*Theta1(:,2:end)); % J != 0
##Theta2_grad=1/m*(Delta_major_2);  %J = 0
##Theta2_grad(:,2:end)=Theta2_grad(:,2:end)+1/m*(lambda*Theta2(:,2:end)); % J != 0

%size(Theta2_grad), %10x26
%size(Theta1_grad), %25x401
%===============================
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end