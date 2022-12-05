import loss.CrossEntropy
clc

function out = loss_fun(X,w,b,y)  % 损失函数 --> 使用交叉熵损失函数
    out = loss.CrossEntropy(pick(net(X,w,b), y+1), 1)
end

% 这个的传入参数是矩阵，所以应该对w(i,j)求偏导数
function grad = grad_w(X, w, b, y, i, j, optional_dx)
    % func 为需要求解梯度的函数, params作为参数传入, argNum标识参数的位置,
    
    % 新赋一个值，防止由于函数修改堆中的数据导致的原始数组改变
    w2 = w;
    l = loss_fun( net(X,w,b), y);
    w2(i,j) = w(i,j) + optional_dx;
    l2 = loss_fun( net(X,w2,b), y);
    grad = (l2 - l)/optional_dx;
end

function grad = grad_b(X, w, b, y, i ,optional_dx)
    b2 = b;
    b2(i) = b(i) + optional_dx ;
    l = loss_fun(X, w, b, y);
    l2 = loss_fun(X, w, b2, y);
    grad = (l2 - l)/optional_dx;
end

function out = accuracy(y_p, y)  % 精确度计算函数
    out = sum(argmax(y_p) == y)/(length(y_p)) ;
end

% 使用matlab自带的softmax函数作为构建模型
function out = net(X, w, b)
% 获取对应函数,使用softmax回归模型做鸢尾花数据训练
    out = softmax(X * w + b');  % 注意w需要进行转置
end

function out = argmax(X)
    s = size(X);
    t = 1: s(2);
    out = sum((max(X,[],2) == X) .* t,2);
end

function out = pick(List ,indexList )
    out = zeros(length(List), 1);
    for i = 1:length(indexList)
        out(i) = List(i, indexList(i));
    end 
end