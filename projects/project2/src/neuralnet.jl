module NeuralNetwork
export NeuralNet, fit!, predict, score, weave!, addLayer!, addSigmoidLayer!, sigmoidactivation, FullyConnected, backpropagation, feedforward, minibatch

using ..Classification: Optimizer, GDContext, SGDContext, NAGDContext, σ, GradientDescent, StochasticGradientDescent, NesterovGradientDescent, partition, postprocess!
import ..Classification: fit!, predict, score, logcrossentropy
using Random: randn
using Statistics: mean

abstract type Layer end

struct FullyConnected <: Layer
    inputsize::Int
    numneurons::Int
    W::Matrix{Float64}
    b::Vector{Float64}
    f::Function
    f′::Function
    function FullyConnected(inputsize, numneurons, transform::Function, derivative::Function)
        # Normal distributed random numbers with N(0, 0.1)
        weights = 1randn(numneurons, inputsize)
        bias = 1randn(numneurons)
        new(inputsize, numneurons, weights, bias, transform, derivative)
    end
end

function newsize(layer::FullyConnected, newsize)::FullyConnected
    FullyConnected(newsize, layer.numneurons, layer.f, layer.f′)
end

struct NeuralNet
    layers::Vector{Layer}
    activations::Vector{Matrix{Float64}}
    learningrate::Float64
end
NeuralNet(layer::Vector{Any}) = NeuralNet(layer, Matrix{Float64}[], 0.1)
NeuralNet(layer::Layer) = NeuralNet([layer], Matrix{Float64}[], 0.1)
NeuralNet() = NeuralNet(Layer[], Matrix{Float64}[], 0.1)

function NeuralNet(neurons::Vector{Int})
    net = NeuralNet()
    if length(neurons) > 0
        # First layer keeps the input 0 until an input is given
        push!(net.layers, FullyConnected(0, neurons[1], sigmoidactivation, ∇σ))
        for numneurons in neurons[2:end]
            addSigmoidLayer!(net, numneurons)
        end
    end
    net
end

function addLayer!(net::NeuralNet, numneurons, transform::Function, derivative::Function)
    inputsize = net.layers[end].numneurons
    layer = FullyConnected(inputsize, numneurons, transform, derivative)
    push!(net.layers, layer)
end

function addSigmoidLayer!(net::NeuralNet, numneurons)
    addLayer!(net, numneurons, sigmoidactivation, ∇σ)
end

function weave!(net, X, y)
    # X has dimension #predictors×#samples
    if length(net.layers) == 0
        push!(net.layers, FullyConnected(size(X, 2), 1, sigmoidactivation, ∇σ))
    else
        net.layers[1] = newsize(net.layers[1], size(X, 1))
    end

    #  Allocate the activation matrices
    #N = length(y)
    #push!(net.activations, zeros(Float64, N, size(X, 2) + 1))
    #for layer in net.layers[1:end-1]
    #    push!(net.activations, zeros(Float64, N, layer.numneurons + 1))
    #end
    # output has no bias
    #push!(net.activations, zeros(Float64, N, 1))
end


# function backpropagation(net::NeuralNet, X::Matrix{T}, y) where T<:Number
    # #penultimate = net.layers[end-1]
    # N = size(y, 1)
    # layer = net.layers[end-1]
    # ŷ = net.activations[end]
    # ∇aᴸC = ŷ .- y
    # δᴸ = ∇aᴸC .* layer.derivative(net.activations[end-1])
    # @show size(δᴸ)
    # @show size(ŷ)
    # @show map(size, net.activations)
    # @show X|>size

    # δ = [similar(a) for a in net.activations]
    # @show length(δ)
    # @show length(net.activations)
    # @show map(size, net.activations)
    # @show map(size, δ)
    # @show size(δᴸ)
    # δ[end-1] .= δᴸ
    # error = ∇aᴸC
    # for l in (length(net.layers)-1):-1:1
        # @show l
        # layer = net.layers[l]
        # a = net.activations[l]
        # @show a |> size
        # @show error |> size
        # δ = error'*a./N
        # @show δ |> size
        # display(δ)
        # W = layer.weights
        # @show layer.derivative(a) |> size
        # @show W |> size
        # @show error |> size
        # error = layer.derivative(a) .* W' * error
    # end
    # for l in (length(net.layers) - 1):-1:1
        # @show l
        # higher = net.layers[l + 1]
        # layer = net.layers[l]
        # @show layer.derivative(net.activations[l]) |> size
        # @show higher.weights
        # @show higher.weights[:, 1:l]' |> size
        # @show δ[end-l] |> size
        # δ[l] = layer.derivative(net.activations[l]) .* higher.weights[:, 1:l]' * δ[l+1]
    # end

    # rate = net.learningrate
    # ∂W = δ[1] * y'./N
    # net.layers[1].weights[:, 1:(end-1)] .-= rate*∂W
    # ∂B = mean(δ[1], 2)
    # for i in 2:(length(net.layers) - 1)
        # ∂C╱∂W = δ[i] * net.activations[i-1]' / N
        # net.layers[i].weights[:, 1:(end-1)] .-= rate*∂C╱∂W
        # ∂C╱∂b = mean(δ[i], 2)
        # net.layers[i].weights[:, end] .-= rate*∂C╱δb
    # end

    # for i in length(net.layers)-1:-1:0
        # @show i
        # @show length(net.layers)
        # layer = net.layers[i]
        # a = net.activations[i]
        # δ = error' * a ./ N

        # @show δ
        # @show error

        # # remove bias if not final layer
        # if layer != length(net.layers) - 1
            # δ = δ[2:end, :]
            # error = error[:, 2:end]
        # end

        # W = layer.weights
        # if i > 1
            # @show layer.derivative(a) |> size
            # @show size(error)
            # @show size(W)
            # @show W' * error |> size
            # error = layer.derivative(a) .* W' * error
        # end

        # W .-= net.learningrate .* δ
    # end
#end

function backpropagation(net::NeuralNet, X, y)
    ∇b = [zeros(Float64, size(layer.b)) for layer in net.layers]
    ∇W = [zeros(Float64, size(layer.W)) for layer in net.layers]

    a = Float64.(X)
    as = [a]
    zs = []
    for layer in net.layers
        z = layer.W * a .+ layer.b
        push!(zs, z)
        a = layer.f(z)
        push!(as, a)
    end
    ∇aᴸC = as[end] .- y

    δ = ∇aᴸC .* net.layers[end].f′(zs[end])
    ∇b[end] .= δ
    ∇W[end] .= δ*as[end-1]'

    for l in 1:(length(net.layers)-1)
        z = zs[end-l]
        sp = net.layers[end-l].f′(z)
        δ = net.layers[end-l+1].W' * δ .* sp
        ∇b[end-l] .= δ
        ∇W[end-l] .= δ * as[end-l-1]'
    end
    ∇b, ∇W
end

function fit!(net::NeuralNet, context::GDContext, X, y)
    optim = GradientDescent(context)
    res = fit!(net, optim, X, y)
    optim
end

function fit!(net::NeuralNet, optim::GradientDescent, X, y)
    # construct the net
    weave!(net, X, y)
    context = optim.context
    optim.loss = zeros(Float64, context.maxiterations)
    for i in 1:context.maxiterations
        GD(net, optim, X, y)
        optim.loss[i] = evaluate(net, X, y)
        optim.iterations += 1
    end
    postprocess!(optim)
end

function GD(net::NeuralNet, optimizer::GradientDescent, X::Matrix{T}, y::Matrix{V}) where {T, V}
    # ∇b, ∇W = backpropagation(net, X, y)
    #@show net.layers[1].b
    N = size(X, 2)
    η = optimizer.context.learningrate
    x′, y′ = 0.0, 0.0
    for sample in 1:size(X, 2)
        x′, y′ = X[:, sample], y[:, sample]
        ∇b, ∇W = backpropagation(net, x′, y′)
        if sample == 1
            #@show ∇b[1]
        end
        for (layer, ∂b, ∂W) in zip(net.layers, ∇b, ∇W)
            layer.b .-= η.*∂b./N
            layer.W .-= η.*∂W./N
        end
    end
end

function fit!(net::NeuralNet, context::SGDContext, X, y; epochs::Int)
    optim = StochasticGradientDescent(context)
    res = fit!(net, optim, X, y, epochs=epochs)
    optim
end

function fit!(net::NeuralNet, optim::StochasticGradientDescent, 
              X::Matrix{T}, y::Matrix{T}; epochs::Int) where T
    numsamples = size(X, 2)
    optim.maxiterations = epochs
    optim.loss = zeros(Float64, optim.context.maxiterations)
    if numsaples % batchsize != 0
        @warn "The minibatch size does not divide number of samples. Leftover $(numsamples%batchsize) dropped"
    end

    @show partitions = partition(numsamples, batchsize)
    for epoch in 1:epochs
        shuffle!(X, y)
        batches = [(X[:, part], y[:, part]) for part in partitions]
        for batch in batches
            minibatch!(net, batch)
        end
        optim.loss[epoch] = evaluate(net, X, y)
        optim.iterations += 1
    end
end

function minibatch!(net::NeuralNet, batch)
    ∇b = [zeros(Float64, size(layer.b)) for layer in net.layers]
    ∇W = [zeros(Float64, size(layer.W)) for layer in net.layers]
    for (x, y) in batch
        Δ∇b, Δ∇W = backpropagation(net, x, y)
        ∇b .+= Δ∇b
        ∇W .+= Δ∇W
    end
    N = length(batche)
    for (layer, ∂b, ∂W) in zip(net.layers, ∇b, ∇W)
        layer.b .-= net.learningrate./N .* ∂b
        layer.W .-= net.learningrate./N .* ∂W
    end
end

function feedforward(net::NeuralNet, X)
    for layer in net.layers
        X = layer.f.(layer.W*X .+ layer.b)
    end
    X
end

function evaluate(net::NeuralNet, X, y)::Float64
    ŷ = predict(net, X)
    sum((y .== ŷ)) / size(X, 2)
end

function logcrossentropy(net::NeuralNet, X, y)
    ŷ = feedforward(net, X)
    loss = logcrossentropy(y, ŷ)
end

function predict(net::NeuralNet, X)::Matrix{Int}
    probabilities = feedforward(net, X)
    return [imax[1] for imax in argmax(probabilities, dims=1)]
end

function ∇σ(x::Array{T})::Array{T} where T<:Number
    x.*sigmoidactivation.(one(T) .- x)
end

function ∇σ(x::T)::T where T<:Number
    x.*sigmoidactivation.(one(T) .- x)
end


function sigmoidactivation(x)
    1.0./(1.0 .+ exp.(-x))
end

function sigmoidactivation(activation, weights)
    1.0./(1.0 .+ exp.(-activation*weights'))
end

function sigmoidactivation!(activation, weights)
    activation .= 1.0./(1.0 .+ exp.(-activation*weights'))
end

function shuffle!(X::Matrix{T} where T, y::Matrix{V} where V)
    n = size(X, 1)
    j = 0
    @inbounds for i in n:-1:2
        @fastmath j = 1 + abs(rand(Int))%n
        swaprows!(X, i, j)
        swaprows!(y, i, j)
        y[i], y[j] = y[j], y[i]
    end
end

function swaprows!(X, i, j)
    for k in 1:size(X, 2)
        X[i, k], X[j, k] = X[j, k], X[i, k]
    end
end

end
