% 主程序，通过调用其他脚本实现算法-----> 这个是单拟合线性回归
clear,clc

% 初始化模型训练参数
iteration_num = 25;
learning_rate = 0.25;   % 定义学习率
batch_size = 80;        % 每次学习的数量

% 原始模型定义
w_true = [2, -3.4];
b_true = 4.2;
X = rand(500,2);
y = sum(X.* w_true, 2) + b_true + 0.01 * rand(500,1);   % 加上rand作为干扰噪声


scatter3(X(:,1),X(:,2),y)

s = size([X,y]);
w = [1.5,-3]';   % 随机生成初始的系数
b = 4;  % 初始时，拟合斜率设置为0


y_p =  net(X,w,b);    % 第一次使用预测结果
disp(join(["iteration 0 :", "loss =" , num2str(squarloss(y_p,y))]))

% 接下来使用小批量随机梯度下降法，每次预估对应的交叉熵损失函数，使交叉熵损失函数达到最小
for iter = 1: 1: iteration_num
% 使用梯度下降算法对模型进行迭代训练
    
    train_data = chooseRandom(batch_size,[X,y]);   % 使用原始的y进行训练
    x_tr = train_data(:, 1:s(2)-1);
    y_tr = train_data(:, s(2));
    for para_index = 1: length(w)
        % ------- caculate gradient ----------
        l1 = loss_func(x_tr, w, b, y_tr);
        w2 = w; w2(para_index) = w(para_index) + 0.005;
        l2 = loss_func(x_tr, w2, b, y_tr);
        grad = (l2-l1)/0.005;
        % ---------使用梯度下降算法-----------------
        w(para_index) = w(para_index) - learning_rate/batch_size * grad;
     
    end
    % ------- 求解梯度 ------
    l1 = loss_func(x_tr, w, b, y_tr);
    b2 = b + 0.005;
    l2 = loss_func(x_tr, w, b2, y_tr);
    grad = (l2-l1)/0.005;
    %-------------------------
    b = b - learning_rate/batch_size * grad;
    
    y_p = net(X, w, b);
    ls = squarloss(y_p, y);   % 注意这个，后面的一项是y,前面的是y^, 是预测值
    disp(join(["iteration",num2str(iter),": ","loss =" , num2str(ls)]))
end
disp("w = ")
disp(w)
disp(join(["b =",num2str(b)]))


% ------------ 使用matlab自带的softmax函数作为构建模型 -----------------
function out = net(X, w, b) 
% 获取对应函数,使用线性回归模型做数据训练模型 (即net = linreg)
    out = sum( X .* w',2) + b;  % 注意w需要进行转置
end

function out = loss_func(X, w, b, y)
    y_p = net(X, w, b);
    out = squarloss(y_p, y);
end

function out = squarloss(y_p, y)
    out = 1/ length(y) * sum(1/2 * (y_p - y).^2);
end
