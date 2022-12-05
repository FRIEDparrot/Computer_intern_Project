% iris预测主程序，通过调用其他脚本实现softmax分类预测算法
clear,clc
import loss.CrossEntropy

% 初始化模型训练参数
iteration_num = 30;
learning_rate = 0.35;   % 定义学习率
batch_size = 10;        % 每次学习的数量
num_input = 4;
num_output = 3;  % 输出的类别有3种

iris = load_iris_data();   % 获取鸢尾花数据集
s = size(iris);
w = rand(num_input,num_output);   % 随机生成初始的系数(4行，3列)
b = zeros(num_output,1);  % 初始时，拟合斜率设置为0

X = iris(:, 1:4);
y = iris(:,5);
y_s = softmax(y);  % y使用softmax函数建立模型
y_p =  net(X,w,b);    % 第一次使用预测结果

disp(join(["iteration 0 :", "loss =" , ...
    num2str(CrossEntropy( pick(y_p, y+1) , 1 )) , "accuracy", num2str(accuracy(net(X,w,b), y+1)) ]))

% 这个的交叉熵损失函数是使用y_p的下标为y    使用-log(pick(y_p, y)) 作为损失函数, 如果乘积越大, 则-log越小
% 由于是pick的y+1,所以对相应的最大列输出有导数，其他的列导数为0  ---> 不会导致非对应列的数值改变
% 接下来使用小批量随机梯度下降法，每次预估对应的交叉熵损失函数，使交叉熵损失函数达到最小

for iter = 1: 1: iteration_num
% 使用梯度下降算法对模型进行迭代训练
    try
        % 如果X_leave已经被定义时
        [train_data, X_leave] = chooseRandom(batch_size, [X, y], X_leave);
        % 使用原始的y进行训练
    catch
        [train_data, X_leave] = chooseRandom(batch_size, [X, y]);    % 其中使用对应的y值进行训练
    end

    x_tr = train_data(:, 1:4);
    y_tr = train_data(:, 5);
    
    % 迭代使用梯度下降方法，通过计算梯度来进行 -----> 注意是求解loss函数对于w的梯度
    % 求解梯度，调节对应的w值，注意得到的部分是 -log ( y_hat.pick(y + 1) ) 
    
    for column = 1: 3
        for line = 1: length(w)
            y_pr =  net(x_tr, w, b(column));   % y_predict
            y_pk = pick(y_pr, y_tr + 1);
            % 由于损失函数是loss
            % grad = grad_w(X, w, b, y, i, j, optional_dx)
            w_grad = grad_w (x_tr, w, b, y_tr, line, column , 0.001);
            w(line, column) = w(line, column) - learning_rate/batch_size * w_grad;
         % 求解梯度
        end
        b(column) = b(column) - learning_rate/batch_size * grad_b(x_tr, w, b, y_tr, column, 0.005 );
    end

    y_p = net(X, w, b);
    % 每一次进行重新预测
    ls = loss.CrossEntropy(y_p(argmax(y_p)),y_s(argmax(y)) );   % 注意这个，后面的一项是y,前面的是y^, 是预测值
    disp(join (["iteration", num2str(iter),":", "loss =" , num2str(CrossEntropy( pick(y_p, y+1) , 1)) , ...
        "accuracy", num2str(accuracy(net(X,w,b), y+1))]))
end

y_p = net(X, w, b);   % 获取最终的预测结果

disp(join( [ "accuracy after iteration:" , num2str(accuracy(net(X, w, b) , y + 1))]))

%----------------------------- plot ---------------------------------------
x_draw = X(:,1);
y_draw = X(:,2);

figure("Name","The Iris Predict")

subplot(1,2,1)
hold on
scatter(x_draw(y == 0),y_draw(y==0),'*r')
scatter(x_draw(y == 1),y_draw(y==1),'*b')
scatter(x_draw(y == 2),y_draw(y==2),'*k')
title("Name", "Origional Result");

subplot(1,2,2)
hold on
title("Name","Predicted Result");
scatter(x_draw(argmax(y_p) == 1), y_draw(argmax(y_p)==1),'*r');
scatter(x_draw(argmax(y_p) == 2), y_draw(argmax(y_p)==2),'*b');
scatter(x_draw(argmax(y_p) == 3), y_draw(argmax(y_p)==3),'*k');

%%-----------------------------functions------------------------------------

function out = loss_fun(X,w,b,y)  % 损失函数 --> 使用交叉熵损失函数
    out = loss.CrossEntropy(pick(net(X,w,b), y+1), 1);
end

% 这个的传入参数是矩阵，所以应该对w(i,j)求偏导数
function grad = grad_w(X, w, b, y, i, j, optional_dx)
    % func 为需要求解梯度的函数, params作为参数传入, argNum标识参数的位置,
    if nargin ~= 7
        warning("You must check");
    end
    % 新赋一个值，防止由于函数修改堆中的数据导致的原始数组改变
    w2 = w;
    l = loss_fun( X,w,b, y);
    w2(i,j) = w(i,j) + optional_dx;
    l2 = loss_fun( X,w2,b, y);
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

function out = pick(List ,indexList)
    out = zeros(length(List), 1);
    for i = 1:length(indexList)
        out(i) = List(i, indexList(i));
    end 
end
% 使用小批量随机梯度下降算法，调用函数来在数据集中选取相应的值
