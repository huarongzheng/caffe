display((sprintf('Program START time:  %s\r',datestr(now))));
clear all; close all;

uClusterNum = 2;
uNumPerCluster = 100;
uClusterSize = uNumPerCluster*ones(1,uClusterNum);
Mu = [[0 0]; [4 5]];
Sigma = cat(3,[1 0; 0 1], [1 0.5; 0.5 1]);
gaussianSrc = zeros(uNumPerCluster,2,uClusterNum);
gaussianSrcramble = zeros(uNumPerCluster*uClusterNum,3);
%figure;
for uCluster = 1:uClusterNum
    gaussianSrc(:,:,uCluster) = mvnrnd(Mu(uCluster,:),Sigma(:,:,uCluster),uClusterSize(uCluster));
    gaussianSrcramble(uNumPerCluster*(uCluster-1)+1:uNumPerCluster*uCluster,1:2) = gaussianSrc(:,:,uCluster);
    if (uCluster == 1)
        format = '+r';
    elseif  (uCluster == 2)
        format = '*b';
    end
%    hold on; plot(gaussianSrc(:,1,uCluster),gaussianSrc(:,2,uCluster),format);
end


temp = gaussianSrcramble;
gaussianSrcramble(randperm(uNumPerCluster*uClusterNum),:) = temp(:,:);
%figure;
for uCluster=1:uClusterNum
    gaussianSrcramble(uNumPerCluster*(uCluster-1)+1:uNumPerCluster*uCluster,3) = uCluster;
    if (uCluster == 1)
        format = '+r';
    elseif  (uCluster == 2)
        format = '*b';
    end
%    hold on; plot(gaussianSrcramble(uNumPerCluster*(uCluster-1)+1:uNumPerCluster*uCluster,1),gaussianSrcramble(uNumPerCluster*(uCluster-1)+1:uNumPerCluster*uCluster,2),format);
end

uIter=1;
uIterTotal = 5;
MSE = zeros(uIterTotal,1);
clusterMean = zeros(uClusterNum,2);
convergenceThreshold = 0.05;
while (uIter<=uIterTotal)
    figure;
    % calc the mean of all clusters
    for uCluster = 1:uClusterNum
        clusterMean(uCluster,:) = mean(gaussianSrcramble(find(gaussianSrcramble(:,3)==uCluster),1:2));
        if (uCluster == 1)
            format = [1 0 0];
        elseif  (uCluster == 2)
            format = [0 0 1];
        end
        hold on; plot(clusterMean(uCluster,1),clusterMean(uCluster,2),'ok','MarkerFaceColor',format);
    end

    % search through all clusters and categorize this sample into the
    % one with the minimum distance.
    for uSample=1:uNumPerCluster*uCluster
        for uCluster = 1:uClusterNum
            difference=gaussianSrcramble(uSample,1:2)-clusterMean(uCluster,:);
            euclidianDistance = difference(1)^2 + difference(2)^2;
            if (uCluster == 1)
                minEuclidianDistance = euclidianDistance;
                gaussianSrcramble(uSample,3) = uCluster;
            end
            if ( euclidianDistance < minEuclidianDistance)
                minEuclidianDistance = euclidianDistance;
                gaussianSrcramble(uSample,3) = uCluster;
            end
        end
        MSE(uIter) = MSE(uIter) + minEuclidianDistance; % sum error
        
        if (gaussianSrcramble(uSample,3) == 1)
            format = '+r';
        elseif  (gaussianSrcramble(uSample,3) == 2)
            format = '*b';
        end
        hold on; plot(gaussianSrcramble(uSample,1),gaussianSrcramble(uSample,2),format);
    end
    
    % mean sum error
    MSE(uIter) = MSE(uIter)/(uNumPerCluster*uCluster);
    
    % converge
    if (uIter>1 && abs(1-MSE(uIter)/MSE(uIter-1))<convergenceThreshold )
        uIter=uIter+1;
        display((sprintf('Converges on the %d iteration when MSE imrpoves less than %d%%\r', uIter, convergenceThreshold*100)));
        break;
    end
    uIter=uIter+1;
end
display(clusterMean);
figure; plot(MSE(1:uIter-1), '-x');title('MSE');
display((sprintf('Program END time:  %s\r',datestr(now))));
