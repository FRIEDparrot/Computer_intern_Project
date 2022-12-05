% 使用小批量随机梯度下降算法，调用函数来在数据集中选取相应的值

function [out, X_leave] = chooseRandom(batch_size, datasets, X_remain)
    if nargin == 2
       X_remain = datasets;
    end 
    % 使用x_leave来获取没有选择的部分

    if (length(X_remain) < batch_size)
        a = X_remain;   % 选取集合X_remain中的全部元素(对于x_remain为[]时，也加上)
        b_new = batch_size - length(X_remain);   % 减去剩余的部分
        b = chooseRandom(b_new, datasets);
        out = [a,b];
        X_leave = [];
    else
        % 对于X_remain没有被选择完毕的情况
        s = size(X_remain);
        index_list = 1:1:length(X_remain);
        list1 = random_shuffle(index_list);

        if s(1)>1 && s(2)>1
            out = X_remain(list1( 1:1:batch_size), :);
            X_p = X_remain;
            X_p(list1(1:1:batch_size), : ) = [];
            X_leave = X_p;
        else
            out = X_remain(list1(1:1:batch_size));   
            X_p = X_remain;
            X_p(list1(1:1:batch_size), : ) = [];
            X_leave = X_p;
        end
    end
end

% use this function to shuffle the array (this array must be 1D)
function out = random_shuffle(list)
    l1 = length(list);
    out = zeros(1, (length(list)));
    for i = 1: 1: l1
        index = ceil(rand() * length(list));
        out(i) = list(index);
        list(index) = [];   % deop the element
    end
end