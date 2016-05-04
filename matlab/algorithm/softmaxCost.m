function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

p = exp(theta*data);
% max: alone dim 1 results in 1*60000; each of the 10 component in 10*60000 
% hypothesis column subtract by 1 max 
p = bsxfun(@minus, p, max(hypothesis, [], 1));
% sum: 1*60000, hypothesis:10*60000
p = bsxfun(@rdivide, p, sum(p, 1));
logHypothesis = log(p);

cost = (-1/numCases)*sum(sum(groundTruth.*logHypothesis,1));

thetagrad = bsxfun(@times, p, sum((groundTruth - groundTruth.*p),1));



% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

