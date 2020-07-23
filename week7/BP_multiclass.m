close all
clear
load group2_train
x = train_data;
x=x';
t = train_labels;
t = t';
t=ind2vec(t);


net = patternnet([500 200 30]); % I tried different no. of layers and it seems 3 is the optimal for this task. The number of neurons per layer should follow a decreasing manner and the exact number is obtained by trial and error. For activation function it seems the default setting works quite well.

net.divideParam.trainRatio = 85/100;
net.divideParam.valRatio = 15/100;
% net.divideParam.testRatio = 15/100;
% net.layers{1}.transferFcn = 'radbas';
% net.layers{2}.transferFcn = 'radbas';
% net.layers{2}.transferFcn = 'softmax';
% net.trainFcn='trainc';
%  net.trainParam.lr = 0.001;
net = train(net,x,t);

y = vec2ind(net(x));

accuracy = sum(y==train_labels')/length(train_labels)

testx = test_data';
testt = test_labels;
testx = reshape(testx,[],length(testx));
t = double(testt)';
testt = ind2vec(t);
y =  vec2ind(net(testx'));
test_accuracy = sum(y==t)/length(t)