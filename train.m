function network = trainNetwork()
    
    % set up aux variables
    columns = 600;
    
    % Concatenate all training matrices to form matrix P
    for i = 1:12
        filename = "P" + int2str(i) + ".mat";
        if i == 1
            P = load(filename).P;
        else
            temp = load(filename).P;
            P = horzcat(P, temp);
        end
    end
    
    load('PerfectArial.mat');   % Target function. Used when the input character is not perfect
    
    % Generate target matrix
    target_out = zeros(10, columns);
    for i = 0 : columns - 1
        target_out(mod(i, 10) + 1, i + 1) = 1; 
    end
    
    target_assoc_mem = repmat(Perfect,1, columns / 10); % For associative memory 

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
    
    net.performParam.lr = 0.01;     %learning rate| default value is 0.01
    net.trainParam.epochs = 100;     %The default is 1000 
                                    %The number of epochs define the number of times that the learning algorithm will work trhough the entire training dataset. One epoch means that each sample in the training dataset has had an opportunity to update the internal model parameters
    net.trainParam.show = 25;       %The default is 25 %show| Epochs between displays
    net.trainParam.goal = 1e-6;     %The default is 0 %goal=objective Performance ggoal
    net.performFcn = 'sse';         %criterion | (Sum Squared error)
    
    net = train(net, P2, target_out);
    sim(net, P);
    
    hardlim_AM_Filter = net;
    save hardlim_AM_Filter;
    
    return
    