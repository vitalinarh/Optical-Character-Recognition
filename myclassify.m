function network = myclassify(data, filled)
    % Extract net
    load('hardlim_Classifier_500.mat');
    % extract result from developed network with given data
    y = sim(net, data);
    % compute the largest elements in each column, as well as the row
    % indices, only row indices matter
    [M, I] = max(y);
    % return only the elements that have been filled
    network = I(filled);       
end    
