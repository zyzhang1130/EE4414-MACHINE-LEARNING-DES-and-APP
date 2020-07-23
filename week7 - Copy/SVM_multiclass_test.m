accuracy1=[];
for j=1:10
    t= test_labels';
for i=1:1996
    if t(i)==j
        t(i)=1;
    else
        t(i)=2;
    end
    
end
t=t';
    [label,scores] = predict(c{j},test_data);
    accuracy = sum(label==t)/length(label);
    accuracy1=[accuracy1,accuracy];
end
mean(accuracy1)