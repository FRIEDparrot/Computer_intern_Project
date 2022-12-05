function grad = autograd(func, param, argNum, optional_dx)
    % func 为需要求解梯度的函数, params作为参数传入, argNum标识参数的位置,
    if nargin == 3
        optional_dx = 0.01 * param(argNum);
    end
    param_str = join(split(string(param)),',');

    eval(join(['y0 = func(', param_str ,');'],''));

    param2 = param;   % 新赋一个值，防止由于函数修改堆中的数据导致的原始数组改变
    param2(argNum) = param(argNum) + optional_dx;

    param2_str = join(split(string(param2)),',');
    eval(join(['y1 = func(', param2_str ,');'],''));
    grad = ( y1 -y0 ) / optional_dx ;
end

% 定义autograd函数求解损失函数对于相应自变量的梯度
% 使用最简单的思路，将自变量增加一个微小的值，再使用函数计算出函数的变化值
% 使用函数变化值 - 自变量的变化值作为相应的点的梯度