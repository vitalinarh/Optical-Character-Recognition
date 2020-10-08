function network = trainNetwork()

    % Choose the amount of test cases for the training dataset
    train_set = menu('Training set: ','500','1000');
    
    if train_set == 1
        columns = 500;
    else
        columns = 1000;
    end
    
    % Concatenate all training matrices to form matrix P
    for i = 1 : columns / 50
        filename = "P" + int2str(i) + ".mat";
        if i == 1
            P = load(filename).P;
        else
            temp = load(filename).P;
            P = horzcat(P, temp);
        end
    end
    
    % Target function. Used on associative memory when the input character is not perfect
    load('PerfectArial.mat');  
    
    % Generate target matrix
    target_out = eye(10, 10);
    I = eye(10, 10);
    for i = 0 : columns / 10 - 2
        target_out = horzcat(target_out, I);
    end
    
    target_assoc_mem = repmat(Perfect, 1, columns / 10); % For associative memory 

    % Filter as Associative memory
    Wp = target_assoc_mem * pinv(P);
    P2 = Wp * P;
    
    net = perceptron; 
                   
    net = configure(net, P2, target_out);
    
    W = rand(10, 256);  % 256 inputs and 10 neurons
    b = rand(10, 1);    % 1 bias value to each neuron, the value will be between 0-1
    
    net.IW{1, 1} = W;
    net.b{1, 1} = b;
    
    % Associative Memory + Classifier
    % hardlim activation function
    
    net.layers{1}.transferFcn = 'hardlim'
    
    net.performParam.lr = 0.01;     % learning rate| default value is 0.01
    net.trainParam.epochs = 1000;   % The default is 1000 
                                    % The number of epochs define the number of times that the learning algorithm will work trhough the entire training dataset. One epoch means that each sample in the training dataset has had an opportunity to update the internal model parameters
    net.trainParam.show = 25;       % The default is 25 %show| Epochs between displays
    net.trainParam.goal = 1e-6;     % The default is 0 %goal=objective Performance ggoal
    net.performFcn = 'sse';         % criterion | (Sum Squared error)
    
    net = train(net, P2, target_out);
    sim(net, P);
    
    hardlim_AM_Filter = net;
    save hardlim_AM_Filter;
    
    return
    