function nnetwork = trainNetwork()

    % Choose the amount of test cases for the training dataset
    train_set = menu('Training set: ','500','1000');
    
    % Choose the architecture for the neural network
    architecture = menu('Architecture:', 'Filter + Classifier', 'Classifier'); 
    
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
    
    % Filter + Classifier
    if architecture == 1
        target_assoc_mem = repmat(Perfect, 1, columns / 10); % For associative memory 

        % Filter as Associative memory
        Wp = target_assoc_mem * pinv(P);
        P2 = Wp * P;
        
        % Choose activation function
        activation_function = menu('Activation Function:', 'Hardlim', 'Linear', 'Sigmoidal');
        
        % Associative Memory + Classifier
        % hardlim activation function
        if activation_function == 1
            net = perceptron;
            net = configure(net, P2, target_out);
            
            net.trainFcn = 'trainc'; % bach is the default
            net.adaptFcn = 'learnp'; % perceptron rule

            net.layers{1}.transferFcn = 'hardlim';
            
            net = init(net);
            
            W = rand(10, 256);  % 256 inputs, 10 neurons
            b = rand(10, 1);    

            net.IW{1, 1} = W;
            net.b{1, 1} = b;

            net.performParam.lr = 0.01;     % learning rate| default value is 0.01
            net.trainParam.epochs = 50;     % The default is 1000 
                                            % The number of epochs define the number of times that the learning algorithm will work trhough the entire training dataset. One epoch means that each sample in the training dataset has had an opportunity to update the internal model parameters
            net.trainParam.show = 25;       % The default is 25 %show| Epochs between displays
            net.trainParam.goal = 1e-6;     % The default is 0 %goal=objective Performance goal
            net.performFcn = 'sse';         % criterion | (Sum Squared error)

            net = train(net, P2, target_out); 

            % Pt = load('P1.mat');
            % a = sim(net, Pt.P);

            % save network on file
            if columns == 500
                hardlim_AM_Filter_500 = net;
                save hardlim_AM_Filter_500;
            else
                hardlim_AM_Filter_1000 = net;
                save hardlim_AM_Filter_1000;
            end
        % linear activation function
        elseif activation_function == 2
            
            net = network;             
            
            net.numInputs = 1;
            net.numLayers = 1;
            
            net.inputs{1}.size = 256;
            net.layers{1}.size = 10;
            
            net.biasConnect(1) = 1;
            net.inputConnect(1,1) = 1;
            net.outputConnect = 1;
            
            net.layers{1}.transferFcn = 'purelin';
            net.layers{1}.initFcn = 'initnw';
            
            net.trainFcn = 'trainb';
            net.trainFcn = 'traingd';
            %net.biases{1}.trainFcn = 'traingd';
            %net.inputWeights{1}.trainFcn = 'traingd'; 
            
            W = rand(10, 256);              % 256 inputs, 10 neurons
            b = rand(10, 1);    

            net.IW{1, 1} = W;
            net.b{1, 1} = b;
            
            net.performParam.lr = 0.01;     % learning rate| default value is 0.01
            net.trainParam.epochs = 50;     % The default is 1000 
                                            % The number of epochs define the number of times that the learning algorithm will work trhough the entire training dataset. One epoch means that each sample in the training dataset has had an opportunity to update the internal model parameters
            net.trainParam.show = 25;       % The default is 25 %show| Epochs between displays
            net.trainParam.goal = 1e-6;     % The default is 0 %goal=objective Performance goal
            net.performFcn = 'sse';         % criterion | (Sum Squared error)
            
            net = init(net);

            net = train(net, P2, target_out); 
            
        % sigmoidal activation function
        else
        end
    % Only Classifier
    else 
        classifier = menu('Type of classifier: ','One Layer','Two Layers', 'Patternet');
        
        % One Layer Classifier
        if classifier == 1
            net = perceptron; 
            net = configure(net, P, target_out);

            W = rand(10, 256);  % 256 inputs and 10 neurons
            b = rand(10, 1);    % 1 bias value to each neuron, the value will be between 0-1

            net.IW{1, 1} = W;
            net.b{1, 1} = b;
            
            % Choose activation function
            activation_function = menu('Activation Function:', 'Hardlim', 'Linear', 'Sigmoidal');

            % hardlim activation function
            if activation_function == 1
                net.trainFcn = 'trainc'; % bach is the default
                net.adaptFcn = 'learnp'; % perceptron rule
                
                net.layers{1}.transferFcn = 'hardlim';

                net.performParam.lr = 0.01;   %learning rate| default value is 0.01
                net.trainParam.epochs = 100;  %The default is 1000 %The number of epochs define the number of times that the learning algorithm will work trhough the entire training dataset. One epoch means that each sample in the training dataset has had an opportunity to update the internal model parameters
                net.trainParam.show = 35;     %The default is 25 %show| Epochs between displays
                net.trainParam.goal = 1e-6;   %The default is 0 goal=objective Performance goal
                net.performFcn = 'sse';       %criterion | (Sum Squared error)

                net = train(net, P, target_out);

                % Pt = load('P1.mat');
                % a = sim(net, Pt.P);

                % save network on file
                if columns == 500
                    hardlim_Classifier_500 = net;
                    save hardlim_Classifier_500;
                else
                    hardlim_Classifier_1000 = net;
                    save hardlim_Classifier_1000;
                end
            % linear activation function
            elseif activation_function == 2
            % sigmoidal activation function
            else
            end
        % Two Layer Classifier
        elseif classifier == 2
            net = network;
            
            net.trainFcn = 'trainc'; % bach is the default
            
            net.numInputs = 1;
            net.numLayers = 2;
            net.biasConnect = [1; 1];
            net.inputConnect(1, 1) = 1;
            net.layerConnect = [0 0; 1 0];
            net.outputConnect = [0 1];
            
            % hidden layer
            net.inputs{1}.size = 256;
            net.layers{1}.size = 100;
            net.layers{1}.transferFcn = 'purelin';
            net.layers{1}.initFcn = 'initnw';
            net.layers{1}.name = 'Hidden';
            
            % output layer
            net.layers{2}.size = 10;
            net.layers{2}.transferFcn = 'tansig';
            net.layers{2}.initFcn = 'initnw';
            
            % Fix number of neurons, more than 10 in principle
            x = 100;
            
            W = rand(x, 256);              % 256 inputs, x neurons
            b = rand(x, 1);    
            
            net.IW{1, 1} = W;
            net.b{1, 1} = b;
            
            W = rand(10, x);              % x inputs, 10 neurons
            b = rand(10, 1);    

            net.LW{2, 1} = W;
            net.b{2, 1} = b;
            
            net.performParam.lr = 0.01;     % learning rate| default value is 0.01
            net.trainParam.epochs = 50;     % The default is 1000 
                                            % The number of epochs define the number of times that the learning algorithm will work trhough the entire training dataset. One epoch means that each sample in the training dataset has had an opportunity to update the internal model parameters
            net.trainParam.show = 25;       % The default is 25 %show| Epochs between displays
            net.trainParam.goal = 1e-6;     % The default is 0 %goal=objective Performance goal
            net.performFcn = 'sse';         % criterion | (Sum Squared error)
            
            net = init(net);
            
            net = train(net, P, target_out); 
               
        % Patternet
        else
            
        end
    end
    
    return
    