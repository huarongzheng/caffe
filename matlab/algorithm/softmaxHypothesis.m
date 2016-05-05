function hyp = softmaxHypothesis(theta,data)
    % phy = e^(theta*data - max(theta*data,1))
    % hyp = phy(i)/sum(phy)
    hyp = theta*data;
    % max: alone dim 1 results in 1*60000; each of the 10 component in 10*60000 
    % hypothesis column subtract by 1 max 
    hyp = bsxfun(@minus, hyp, max(hyp, [], 1));
    hyp = exp(hyp);
    % sum: 1*60000, hypothesis:10*60000
    hyp = bsxfun(@rdivide, hyp, sum(hyp, 1));
end