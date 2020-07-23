clear all
% rng(3);
X = zeros(5,5,5);

X(:,:,1) = [0 1 1 0 0;
            0 0 1 0 0;
            0 0 1 0 0;
            0 0 1 0 0;
            0 1 1 1 0];
X(:,:,2) = [1 1 1 1 0;
            0 0 0 0 1;
            0 1 1 1 0;
            1 0 0 0 0;
            1 1 1 1 1];
X(:,:,3) = [1 1 1 1 0;
            0 0 0 0 1;
            0 1 1 1 0;
            0 0 0 0 1;
            1 1 1 1 0];
X(:,:,4) = [0 0 0 1 0;
            0 0 1 1 0;
            0 1 0 1 0;
            1 1 1 1 1;
            0 0 0 1 0];
X(:,:,5) = [1 1 1 1 1;
            1 0 0 0 0;
            1 1 1 1 0;
            0 0 0 0 1;
            1 1 1 1 0];
D = [1 0 0 0 0;
    0 1 0 0 0;
    0 0 1 0 0;
    0 0 0 1 0;
    0 0 0 0 1];

W1 = 2*rand(20,25) - 1;
W2 = 2*rand(20,20) - 1;
W3 = 2*rand(20,20) - 1;
W4 = 2*rand(5,20) - 1;

for epoch = 1:20000
[W1, W2, W3, W4] = DeepDropout(W1, W2, W3, W4, X, D);
end

N=5;
for k =1:N
    x = reshape(X(:,:,k),25, 1);
    v1 = W1*x;
    y1 = Sigmoid(v1);
    
    v2 = W2*y1;
    y2 = Sigmoid(v2);
    
    v3 = W3*y2;
    y3 = Sigmoid(v3);
    
    v = W4*y3;
    y = Softmax(v)
end



function [W1, W2, W3, W4] = DeepDropout(W1, W2, W3, W4, X, D)
    alpha = 0.01;

    N = 5;
    for k = 1:N
        x = reshape(X(:,:,k), 25 , 1);
        v1 = W1*x;
        y1 = Sigmoid(v1);
        y1 = y1.*Dropout(y1, 0.2);
        
        v2 = W2*y1;
        y2 = Sigmoid(v2);
        y2 = y2.*Dropout(y2, 0.2);
        
        
        v3 = W3*y2;
        y3 = Sigmoid(v3);
        y3 = y3.*Dropout(y3, 0.2);
        
        v = W4*y3;
        y = Softmax(v);
        
        d = D(k, :)';
        
        e = d - y;
        delta = e;
        
        e3 = W4'*delta;
        delta3 =  y3.*(1-y3).*e3;
        
        e2 = W3'*delta3;
        delta2 =  y2.*(1-y2).*e2;
        
        e1 = W2'*delta2;
        delta1 =  y1.*(1-y1).*e1;
        
        dW4 = alpha*delta*y3';
        W4 = W4 + dW4;
        
        dW3 = alpha*delta3*y2';
        W3 = W3 + dW3;
        
        dW2 = alpha*delta2*y1';
        W2 = W2 + dW2;
        
        dW1 = alpha*delta1*x';
        W1 = W1 + dW1;
    end
end

function y = Softmax(x)
    ex = exp(x);
    y = ex / sum(ex);
end

function y = Sigmoid(x)
y=1./(1+exp(-x));
end

function ym = Dropout(y, ratio)
    [m,n] = size(y);
    ym = zeros(m,n);

    num = round(m*n*(1-ratio));
    idx = randperm(m*n, num);
    ym(idx) = 1/(1-ratio);
end