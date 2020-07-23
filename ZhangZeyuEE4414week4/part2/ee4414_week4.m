close all
clear

[x,t] = digitTrain4DArrayData;
%imshow(x(:,:,:,randi(length(t))));
trainx = reshape(x,[],5000);
t = double(t)';
traint=ind2vec(t);

net = patternnet([300 150 20]); % I tried different no. of layers and it seems 3 is the optimal for this task. The number of neurons per layer should follow a decreasing manner and the exact number is obtained by trial and error. For activation function it seems the default setting works quite well.


net = train(net,trainx,traint);

y = vec2ind(net(trainx));

accuracy = sum(y==t)/length(t)



[testx,testt] = digitTest4DArrayData;
%testx = reshape(testx,[],length(testx));
t = double(testt)';
testt = ind2vec(t);


y =  vec2ind(net(testx));
test_accuracy = sum(y==t)/length(t)