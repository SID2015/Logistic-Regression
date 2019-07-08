function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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


h = X*theta;

h_sigm = sigmoid(h);

left_summation = ((y)')*(log(h_sigm));

right_summation = ((1-y)')*(log(1-h_sigm));

summation_cost = left_summation + right_summation ;

summation_cost = -((1/m)*(summation_cost));

theta(1) = 0;
regularization = (lambda/(2*m))*((theta')*(theta));

J = summation_cost + regularization ;


grad = ((1/m)*(X')*(h_sigm - y)) + (lambda/m)*(theta);


% =============================================================

end
