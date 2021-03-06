%% CS294A/CS294W Softmax Exercise 

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  softmax exercise. You will need to write the softmax cost function 
%  in softmaxCost.m and the softmax prediction function in softmaxPred.m. 
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%  (However, you may be required to do so in later exercises)

%%======================================================================
%% STEP 0: Initialise constants and parameters
%
%  Here we define and initialise some constants which allow your code
%  to be used more generally on any arbitrary input. 
%  We also initialise some parameters used for tuning the model.
%display(sprintf('Program START time:  %s\r',datestr(now)));
clc; clear all; close all;

inputSize = 28 * 28; % Size of input vector (MNIST images are 28x28)
numClasses = 10;     % Number of classes (MNIST images fall into 10 classes)

lambda = 1e-4; % Weight decay parameter

numpatches = inf;
%%======================================================================
%% STEP 1: Load data
%
%  In this section, we load the input and output data.
%  For softmax regression on MNIST pixels, 
%  the input data is the images, and 
%  the output data is the labels.
%

% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as 
% train-images.idx3-ubyte / train-labels.idx1-ubyte

inputData = loadMNISTImages('./mnist/train-images-idx3-ubyte',numpatches);
labels = loadMNISTLabels('./mnist/train-labels-idx1-ubyte',numpatches);
labels(labels==0) = 10; % Remap 0 to 10

numpatches = size(inputData,2);
assert(size(labels,1) == numpatches, 'Mismatch in label count');


% For debugging purposes, you may wish to reduce the size of the input data
% in order to speed up gradient checking. 
% Here, we create synthetic dataset using random data for testing

DEBUG = false; % Set DEBUG to true when debugging.
if DEBUG
    assert(numpatches>1000, 'too many images to show for debug, try numpatches < 1000');
%    inputSize = 8;
%    inputData = randn(inputSize, numpatches);
%    labels = randi(numClasses, numpatches, 1);
    
    % We are using display_network from the autoencoder code
    display_network(inputData(:,1:numpatches)); % Show the first 100 images
    %disp(labels(1:numpatches));
end

% Randomly initialise theta
theta = 0.005 * randn(numClasses * inputSize, 1);

%%======================================================================
%% STEP 2: Implement softmaxCost
%
%  Implement softmaxCost in softmaxCost.m. 

[cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, inputData, labels);
                                     
%%======================================================================
%% STEP 3: Gradient checking
%
%  As with any learning algorithm, you should always check that your
%  gradients are correct before learning the parameters.
% 

if DEBUG
    numGrad = computeNumericalGradient( @(x) softmaxCost(x, numClasses, ...
                                    inputSize, lambda, inputData, labels), theta);

    % Use this to visually compare the gradients side by side
    % disp([numGrad grad]); 
    figure; plot(numGrad-grad);
    
    % Compare numerically computed gradients with those computed analytically
    diff = norm(numGrad-grad)/norm(numGrad+grad);
    display(sprintf('norm diff = %e\r',diff)); %disp(diff); 
    % The difference should be small. 
    % In our implementation, these values are usually less than 1e-7.

    % When your gradients are correct, congratulations!
end

%%======================================================================
%% STEP 4: Learning parameters
%
%  Once you have verified that your gradients are correct, 
%  you can start training your softmax regression code using softmaxTrain
%  (which uses minFunc).

% refer to comment in ./minFunc/minFunc.m
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % softmaxCost.m satisfies this.
options.maxIter = 100;
options.useMex = false;
options.Display = 'off'; % Level [ off | final | (iter) | full | excessive ]
softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
                            inputData, labels, options);
display_network(softmaxModel.optTheta'); 

% Although we only use 100 iterations here to train a classifier for the 
% MNIST data set, in practice, training for more iterations is usually
% beneficial.

%%======================================================================
%% STEP 5: Testing
%
%  You should now test your model against the test images.
%  To do this, you will first need to write softmaxPredict
%  (in softmaxPredict.m), which should return predictions
%  given a softmax model and the input data.

inputData = loadMNISTImages('./mnist/t10k-images-idx3-ubyte',inf);
labels = loadMNISTLabels('./mnist/t10k-labels-idx1-ubyte',inf);
labels(labels==0) = 10; % Remap 0 to 10


% You will have to implement softmaxPredict in softmaxPredict.m
[pred] = softmaxPredict(softmaxModel, inputData);

acc = mean(labels(:) == pred(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);
assert(acc * 100>92.2, 'accuracy > 92.200%, normally it goes up to 92.6%');
%display(sprintf('Program END time:  %s\r',datestr(now)));
% Accuracy is the proportion of correctly classified images
% After 100 iterations, the results for our implementation were:
%
% Accuracy: 92.200%
%
% If your values are too low (accuracy less than 0.91), you should check 
% your code for errors, and make sure you are training on the 
% entire data set of 60000 28x28 training images 
% (unless you modified the loading code, this should be the case)
