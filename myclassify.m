function network = myclassify(data, filled)
    % Extract net
<<<<<<< HEAD
    load('hardlim_Classifier_500.mat');
=======
    load('hardlim_AM_Filter_1000.mat');
>>>>>>> f3a652af4d7b8316849515e6a133e07fd695bc47
    % extract result from developed network with given data
    y = sim(net, data);
    % compute the largest elements in each column, as well as the row
    % indices, only row indices matter
    [M, I] = max(y);
    network = I(filled);       
end    
