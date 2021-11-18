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

% a1 = X
X = [ones(m,1) X]; %add bias unit 5000x401
a1 = X;
z2 = a1 * Theta1'; % (5000x401) x (401x25)
a2 = sigmoid(z2); % 5000x25 

a2 = [ones(m,1) a2]; %add bias unit 5000x26
z3 = a2 * Theta2'; % (5000x26) x (26x10)
a3 = sigmoid(z3); % 5000x10

%untuk label yang hanya nilainya 0 dan 1 dimensi K
I = eye(num_labels); %identity 10x10
Y_matrix = I(y, :);

%compute cost function
firstJ = -(Y_matrix .* log(a3));
secondJ = (1-Y_matrix) .* (log(1-a3));

J = (1/m) * sum(sum(firstJ-secondJ));

%regularizataion
%note that we should not the corespond to bias unit
%first column of each martrices Theta1 & Theta2 is corespond to bias unit
th1 = Theta1(:, 2:end);
th2 = Theta2(:, 2:end);
firstReg = sum(sum(th1 .^ 2));
secondReg = sum(sum(th2 .^ 2));
reg = (lambda / (2 * m)) * (firstReg + secondReg);

J = J + reg;

%
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
% coro cepat
%delta_3 = a3 - Y_matrix; % 5000 x 10

for t=1:m
  
  % Step 1
  a_1 = a1(t, :); % 1 x 401
  z_2 = Theta1 * a_1';  % (25x401) x (401*1)
  a_2 = sigmoid(z_2); % (25*1)
  
  a_2 = [1; a_2]; % add bias term, 26x1
  z_3 = Theta2 * a_2; % (10x26) x (26x1)
  a_3 = sigmoid(z_3); % 10 x 1
  
  % step 2
  delta_3 = a_3 - Y_matrix(t, :)'; % 10x1
  
  % step 3
  z_2 = [1; z_2]; % 26x1
  delta_2 = (Theta2' * delta_3) .* sigmoidGradient(z_2); %(26x10)x(10x1)
  %delta_2 26x1
  
  % step 4
  delta_2 = delta_2(2:end); %exclude bias 25x1
  
  Theta2_grad = Theta2_grad + delta_3 * a_2'; % (10x1)x(1x26)
  Theta1_grad = Theta1_grad + delta_2 * a_1; % (25x1)x(1x401)

end;

% step 5
Theta2_grad = (1/m) * Theta2_grad; % 10x26
Theta1_grad = (1/m) * Theta1_grad; % 25x401
  
  

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda/m) * Theta1(:, 2:end);

Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda/m) * Theta2(:, 2:end);




















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
