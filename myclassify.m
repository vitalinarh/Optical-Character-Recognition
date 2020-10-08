function network = myclassify(data, filled)
    % Extract net
    load('hardlim_AM_Filter_1000.mat');
    % extract result from developed network with given data
    y = net(data);
    % compute the largest elements in each column, as well as the row
    % indices, only row indices matter
    [M, I] = max(y);
    network = I(filled);       
end    
