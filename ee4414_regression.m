clear
clc

numFeatures = 3;
numNeurons = [700 500]; 
%numNeurons = 30;

% prepare dataset
switch numFeatures
    case 1
        x = 0:2*pi/99:2*pi;
        tempt = sin(x);
        t = awgn(tempt,20);
    case 2
        x = 0:2*pi/99:2*pi;
        x2 = 2*x;
        tempt = sin(x) + cos(x2);
        x = [x;x2];
        t = awgn(tempt,20); % add noise with snr=20
    case 3
        x = 0:2*pi/99:2*pi;
        x2 = 2*x;
        x3 = 3*x;
        tempt = sin(x) + cos(x2) + tan(x3);
        x = [x;x2;x3];
        t = awgn(tempt,20);
    otherwise
        x = 0:2*pi/99:2*pi;
        tempt = sin(x);
end


%[x,t] = abalone_dataset;
            
% init the network
net = fitnet(numNeurons);

% modify the parameters (your code here)

 net.layers{1}.transferFcn = 'logsig';
 net.layers{2}.transferFcn = 'radbas';
 net.trainFcn='traincgb';
 net.trainParam.lr = 0.001;





% train the network
net = train(net,x,t);

% retrive the prediction
y = net(x);

% plot the truth and prediction
if strcmp(net.name,'Function Fitting Neural Network')
    plot(t)
    hold on
    plot(y)
%     hold on
%     plot(tempt) 
%     legend('truth','prediction','truth without noise');
    legend('truth','prediction');
elseif strcmp(net.name,'Pattern Recognition Neural Network')
    [T,tI] = max(t);
    [Y,yI] = max(y);
    [~,sI] = sort(tI);
    sI=sI(1:10);
    subplot(1,2,1)
    plot(tI(sI),'go')
    title('truth')
    subplot(1,2,2)
    plot(yI(sI),'ro')
    title('prediction')
else 
end

    
    
    
    
    
    