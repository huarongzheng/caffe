function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, a1)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data (a1): Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);


% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));


%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

numCases = size(a1, 2);

% forward
for layer = 2:3
  if (layer == 2)
      W=W1;
      b=b1;
      a=a1;
  elseif  (layer == 3)
      W=W2;
      b=b2;
  end

  z = W*a + repmat(b,1,size(a,2));
  a = sigmoid(z);
  
  if (layer == 2)
      a2=a;
  elseif  (layer == 3)
      a3=a;
  end
end

sparsity = (1/numCases)*sum(a2,2);
error = a3 - a1; %a3==a
[klDivergence, klDivergenceDev] = KL(sparsity, sparsityParam);
cost = (1/numCases)*sum(sum(  0.5*(error.^2)   ,1)) + (lambda/2)*sum(sum(theta.*theta)) + beta*sum(klDivergence);

%backpropagation
delta3 = error.*a3.*(1-a3);
delta2 = (W2'*delta3 + beta*repmat(klDivergenceDev,1,size(delta3,2))).*a2.*(1-a2);
delta1 = W1'*delta2.*a1.*(1-a1);

W2grad = (1/numCases)*delta3*a2' + lambda*W2;
W1grad = (1/numCases)*delta2*a1' + lambda*W1;
b2grad = (1/numCases)*sum(delta3,2) + lambda*b2;
b1grad = (1/numCases)*sum(delta2,2) + lambda*b1;

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:); W2grad(:); b1grad(:); b2grad(:)];
fflush(stdout); % flush all previous msg out especially per iter info in minFunc
end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

function [klDivergence, klDivergenceDev] = KL(sparsity, sparsityParam)
    diff0 = sparsityParam./sparsity;
    diff1 = (1-sparsityParam)./(1-sparsity);
    klDivergence = sparsityParam*log(diff0)+(1-sparsityParam)*log(diff1);
    klDivergenceDev = -diff0+diff1;
end
