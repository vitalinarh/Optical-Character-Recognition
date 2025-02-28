function network = myclassify(data, filled)

    dataset = menu ('Data set:', '500', '1000');

    nn_architecture = menu('Achitecture:', 'Filter + Classfier', 'Classifier');
    
    if dataset == 1
        n = 500;
    else
        n = 1000;
    end
    
    if nn_architecture == 1
        activation_function = menu('Activation Function:', 'Hardlim', 'Linear', 'Sigmoid');
        
        if activation_function == 1
            load("hardlim_AM_Filter_" + int2str(n) + ".mat");
        elseif activation_function == 2
            load("linear_AM_Filter_" + int2str(n) + ".mat");
        else
            load("sigmoid_AM_Filter_" + int2str(n) + ".mat");
        end
    else
        classifier = menu('Type of classifier: ','One Layer','Two Layers', 'Patternet'); 
        if classifier == 1
            activation_function = menu('Activation Function:', 'Hardlim', 'Linear', 'Sigmoid');
            if activation_function == 1
            load("hardlim_Classifier_" + int2str(n) + ".mat");
            elseif activation_function == 2
                load("linear_Classifier_" + int2str(n) + ".mat");
            else
                load("sigmoid_Classifier_" + int2str(n) + ".mat");
            end
        elseif classifier == 2
            load("two_layer_" + int2str(n) + ".mat");
        else
            load("patternet_" + int2str(n) + ".mat");
        end
    end 
   
    % extract result from developed network with given data
    y = sim(net, data);
    % compute the largest elements in each column, as well as the row
    % indices, only row indices matter
    [M, I] = max(y);
    % return only the elements that have been filled
    network = I(filled);       
end