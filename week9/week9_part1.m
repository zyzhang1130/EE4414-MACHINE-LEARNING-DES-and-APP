clear all
rng(3);
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

W1 = 2*rand(50,25) - 1;
W2 = 2*rand(5,50) - 1;

for epoch = 1:10000
[W1, W2] = Multiclass(W1, W2, X, D);
end

N=5;
for k =1:N
    x = reshape(X(:,:,k),25, 1);
    v1 = W1*x;
    y1 = Sigmoid(v1);
    v = W2*y1;
    y = Softmax(v)
end

function [W1, W2] = Multiclass(W1, W2, X, D)
    alpha = 0.9;

    N = 5;
    for k = 1:N
        x = reshape(X(:,:,k), 25 , 1);
        d = D(k, :)';

        v1 = W1*x;
        y1 = Sigmoid(v1);
        v = W2*y1;
        y = Softmax(v);

        e = d - y;
        delta = e;

        e1 = W2'*delta;
        delta1 = y1.*(1-y1).*e1;

        dW1 = alpha*delta1*x';
        W1 = W1 + dW1;

        dW2 = alpha*delta*y1';
        W2 = W2 + dW2;

    end
end

function y = Softmax(x)
    ex = exp(x);
    y = ex / sum(ex);
end

function y = Sigmoid(x)
y=1./(1+exp(-x));
end