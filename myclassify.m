function network = myclassify(data, filled)
    % Extract net
<<<<<<< HEAD
    load('hardlim_Classifier_500.mat');
=======
    load('hardlim_AM_Filter_1000.mat');
>>>>>>> bea727cef043ceedc325e118fe5b5370849f4828
    % extract result from developed network with given data
    y = sim(net, data);
    % compute the largest elements in each column, as well as the row
    % indices, only row indices matter
    [M, I] = max(y);
    % return only the elements that have been filled
    network = I(filled);       
end    
