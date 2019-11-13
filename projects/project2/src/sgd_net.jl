import .Classification: fit!
using .Classification: SGDContext, StochasticGradientDescent
using  .NeuralNetwork: NeuralNet, backpropagation, evaluate!

function fit!(net::NeuralNet, context::SGDContext, X, y, labels)
    optim = StochasticGradientDescent(context)
    res = fit!(net, optim, X, y, labels)
    optim
end

function fit!(net::NeuralNet, optim::StochasticGradientDescent, 
              X::Matrix{<:Real}, y::Matrix{<:Real}, labels)
    weave!(net, X, y)
    numsamples = size(X, 2)
    optim.loss = zeros(Float32, optim.context.maxiterations)
    batchsize = optim.context.batchsize
    if numsamples % batchsize != 0
        @warn "The minibatch size does not divide number of samples. Leftover $(numsamples%batchsize) dropped"
    end

    partitions = partition(numsamples, batchsize)
    for epoch in 1:optim.context.maxiterations
        shuffle!(X, y, labels)
        for batch in partitions
            minibatch!(net, optim, X, y, batch)
        end
        optim.iterations += 1
        evaluate!(net, optim, X, labels)
    end
end

function minibatch!(net::NeuralNet, optim::StochasticGradientDescent, X, y, batch)
    ∇b = [zeros(Float32, size(layer.b)) for layer in net.layers]
    ∇W = [zeros(Float32, size(layer.W)) for layer in net.layers]
    for i in batch
        Δ∇b, Δ∇W = backpropagation(net, X[:, i], y[:, i])
        ∇b .+= Δ∇b
        ∇W .+= Δ∇W
    end
    η = optim.context.learningrate
    B = length(batch)
    for (layer, ∂b, ∂W) in zip(net.layers, ∇b, ∇W)
        layer.b .-= η./B .* ∂b
        layer.W .-= η./B .* ∂W
    end
end

function shuffle!(X::Matrix{<:Real}, y::Matrix{<:Real})
    n = size(X, 2)
    j = 0
    for i in n:-1:2
        @fastmath j = 1 + abs(rand(Int))%n
        swapcols!(X, i, j)
        swapcols!(y, i, j)
    end
end

function shuffle!(X::Matrix{<:Real}, y::Matrix{<:Real}, z)
    n = size(X, 2)
    j = 0
    for i in n:-1:2
        @fastmath j = 1 + abs(rand(Int))%n
        swapcols!(X, i, j)
        swapcols!(y, i, j)
        swapcols!(z, i, j)
    end
end

function swapcols!(X, i, j)
    for k in 1:size(X, 1)
        X[k, i], X[k, j] = X[k, j], X[k, i]
    end
end
