%preprocessing of the photos
path=pwd;
folder=strcat(path,'\face');
I=dir(fullfile(folder));
for k=1:numel(I)
    try
        filename=fullfile(folder,I(k).name);
        I2{k}=imread(filename);
        I2{k}=rgb2gray(I2{k});          %change RGB images of grey scale images
        I2{k}=imresize(I2{k}, [64 64]); %specify the resolution of the images
    catch
    end
end
I2=[I2(1,3:7),I2(1,13:47)];
train1=I2(1,1:10);
train2=I2(1,21:30);
test1=I2(1,11:20);
test2=I2(1,31:40);
trainxx=[train1,train2];
testxx=[test1,test2];
traint=zeros(2,2);
traint(1,1:10)=1;
traint(2,11:20)=1;
testt=zeros(2,2);
testt(1,1:10)=1; 
testt(2,11:20)=1;   %this traint and testt have the same format as asked in the lab manual (one-hot encoding, however, not very useful in the training process. Hence, new labels are created later.
trainx=[];
testx=[];
[m,n]=size(trainxx{1});
for i=1:20
    trainx=[trainx,reshape(trainxx{i},[m^2,1])];
    testx=[testx,reshape(testxx{i},[m^2,1])];
end


%creating labels
[m,n]=size(trainx);
traint=zeros(1,20);
traint(1:10)=1;
traint(11:20)=2;
testt=zeros(1,20);
testt(1:10)=1;
testt(11:20)=2;
traint2=traint;
traint=ind2vec(traint);



trainx=double(trainx);
testx=double(testx);
net = patternnet([128 64 10]); % I tried different no. of layers and it seems 3 is the optimal for this task. The number of neurons per layer should follow a decreasing manner and the exact number is obtained by trial and error. For activation function it seems the default setting works quite well.

net = train(net,trainx,traint);

%traing accuracy
y1 = vec2ind(net(trainx));
train_accuracy = sum(y1==traint2)/length(traint2)

%testing accuracy
y2 =  vec2ind(net(testx));
test_accuracy = sum(y2==testt)/length(testt)
