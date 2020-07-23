load group2_train
c={};
for j=1:10
x = train_data;
t= train_labels;
for i=1:size(t)
    if t(i)==j
        t(i)=1;
    else
        t(i)=2;
    end
end
%x=x';
%t=vec2ind(t');
%t=t';

CVP = cvpartition(t, 'Holdout', 0.2);
trainingIdx = training(CVP);
testIdx = test (CVP);
% trainx=x(1:8000,:);
% traint=t(1:8000,:);
cl = fitcsvm(x(trainingIdx,:),t(trainingIdx),...
    'KernelFunction','rbf',...
    'BoxConstraint',Inf,...
    'ClassNames',[1,2]);
c{j}=cl
[label,scores] = predict(cl,x(testIdx,:));
accuracy = sum(label==t(testIdx))/length(label)
end