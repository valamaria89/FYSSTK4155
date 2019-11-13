import .Classification: fit!
using .Classification: GDContext, GradientDescent
using  .NeuralNetwork: NeuralNet, backpropagation, evaluate!


function fit!(net::NeuralNet, context::GDContext, X, y, labels)
    optim = GradientDescent(context)
    res = fit!(net, optim, X, y, labels)
    optim
end

function fit!(net::NeuralNet, optim::GradientDescent, X, y, labels)
    # construct the net
    weave!(net, X, y)
    context = optim.context
    optim.loss = zeros(Float32, context.maxiterations)
    for i in 1:context.maxiterations
        GDs(net, optim, X, y)
        optim.iterations += 1
        evaluate!(net, optim, X, labels)
    end
    postprocess!(optim)
end

function GDs(net::NeuralNet, optimizer::GradientDescent, X::Matrix{<:Real}, y::Matrix{<:Integer})
    N = size(X, 2)
    η = optimizer.context.learningrate
    ∇b = [zeros(Float32, size(layer.b)) for layer in net.layers]
    ∇W = [zeros(Float32, size(layer.W)) for layer in net.layers]
    for sample in 1:size(X, 2)
        b, W = backpropagation(net, X[:, sample],
                               y[:, sample])
        ∇b .+= b
        ∇W .+= W
    end
    for (layer, ∂b, ∂W) in zip(net.layers, ∇b, ∇W)
        layer.b .-= η.*∂b ./N
        layer.W .-= η.*∂W ./N
    end
end

