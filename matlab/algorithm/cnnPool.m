function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(featureNum, imageNum, poolRow, poolCol)
%     

numImages = size(convolvedFeatures, 2);
numFeatures = size(convolvedFeatures, 1);
convolvedDim = size(convolvedFeatures, 3);
convolvedPoolDim = floor(convolvedDim / poolDim);

pooledFeatures = zeros(numFeatures, numImages, convolvedPoolDim, convolvedPoolDim);

% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   numFeatures x numImages x (convolvedDim/poolDim) x (convolvedDim/poolDim) 
%   matrix pooledFeatures, such that
%   pooledFeatures(featureNum, imageNum, poolRow, poolCol) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region 
%   (see http://ufldl/wiki/index.php/Pooling )
%   
%   Use mean pooling here.
% -------------------- YOUR CODE HERE --------------------
for poolRow = 0:convolvedPoolDim-1
    for poolCol = 0:convolvedPoolDim-1
        idx3 = poolRow*poolDim+1:poolRow*poolDim+poolDim;
        idx4 = poolCol*poolDim+1:poolCol*poolDim+poolDim;
        subMatrix = convolvedFeatures(:,:,idx3,idx4);
        pooledFeatures(:,:,poolRow+1,poolCol+1) = mean(mean(subMatrix,3),4);
    end  
end

end

