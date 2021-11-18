function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
n = size(X, 2);
reg = (lambda / (2*m)) * sum(theta(2:n) .^ 2);
hx = sigmoid(X*theta);
leftJ = -y' * log(hx);
rightJ = (1-y)' * log(1-hx);
J = ((1/m) * (leftJ - rightJ)) + reg;

%grad(1) = (1/m) * ((hx-y)' * X);
%grad(2:n) = ((1/m) * (hx-y)' * X) + (lambda/m)*theta;
theta(1) = 0;
grad = ((1/m) * (hx-y)' * X) + (lambda/m)*theta;



% =============================================================

end
