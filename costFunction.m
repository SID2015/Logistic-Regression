function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% We need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== OUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               We should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

h = X*theta;

h_sigm = sigmoid(h);

left_summation = ((y)')*(log(h_sigm));

right_summation = ((1-y)')*(log(1-h_sigm));

summation_cost = left_summation + right_summation ;

J = -((1/m)*(summation_cost));


grad = (1/m)*(X')*(h_sigm - y);


% =============================================================

end
