close all
clear
clc

[x,t] = iris_dataset;
fprintf('%d samples, %d input features, %d classes.\n', ...
    size(x,2), size(x,1), size(t,1))

tsnex = tsne(x');
subplot(2,2,[1,2]);
gscatter(tsnex(:,1),tsnex(:,2),t'); %plot ground truth


net = patternnet;
net = train(net,x,t);
y = net(x);

y_ind = vec2ind(y);
subplot(2,2,[3,4]);
%figure
gscatter(tsnex(:,1),tsnex(:,2),y_ind'); %plot prediction


desktop = com.mathworks.mde.desk.MLDesktop.getInstance;
desktop.restoreLayout('Default');