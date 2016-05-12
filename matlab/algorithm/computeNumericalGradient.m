function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros
numgrad = zeros(size(theta));

%% ---------- YOUR CODE HERE --------------------------------------
% Instructions: 
% Implement numerical gradient checking, and return the result in numgrad.  
% (See Section 2.3 of the lecture notes.)
% You should write code so that numgrad(i) is (the numerical approximation to) the 
% partial derivative of J with respect to the i-th input argument, evaluated at theta.  
% I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
% respect to theta(i).
%                
% Hint: You will probably want to compute the elements of numgrad one at a time. 

epsilon = 10e-4;

thetaDim = size(theta,1);
for pos = 1:thetaDim
    fprintf('computeNumericalGradient: theta element %f%%\n', 100*(pos/thetaDim));
    fflush(stdout);
    thetaTemp = theta;
    thetaTemp(pos) = thetaTemp(pos) + epsilon;
    JPlus = J(thetaTemp);
    thetaTemp(pos) = thetaTemp(pos) - 2*epsilon; % theta(pos) - epsilon
    JMinus = J(thetaTemp);
    numgrad(pos) = (JPlus - JMinus) / (2*epsilon);
end

%% ---------------------------------------------------------------
end
