[x,t] = cancer_dataset;
x=x';
t=vec2ind(t);
t=t';

CVP = cvpartition(t, 'Holdout', 0.2);
trainingIdx = training(CVP);
testIdx = test (CVP);

cl = fitcsvm(x(trainingIdx,:),t(trainingIdx),...
    'KernelFunction','rbf',...
    'BoxConstraint',Inf,...
    'ClassNames',[1,2]);

[label,scores] = predict(cl,x(testIdx,:));
accuracy = sum(label==t(testIdx))/length(label)