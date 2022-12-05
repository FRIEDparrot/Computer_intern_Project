function out = sgd(params, learning_rate, batch_size)

    for i = 1:1:length(params)
        param = params[i];     % gain the i^th parameter
        param[:] = param - lr * grad(param)/batch_size;
    end

end