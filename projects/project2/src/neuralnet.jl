module NeuralNetwork
export NeuralNet, fit!, predict, score, weave!, addLayer!, addSigmoidLayer!, FullyConnected, backpropagation, feedforward, minibatch, evaluate, addSoftmaxLayer!

using ..Classification: Optimizer, GDContext, SGDContext, NAGDContext, GradientDescent, StochasticGradientDescent, NesterovGradientDescent, partition, postprocess!
import ..Classification: fit!, predict, score, logcrossentropy
using Random: randn
using Statistics: mean
using NPZ: npzwrite, npzread

abstract type Layer end

struct FullyConnected <: Layer
    inputsize::Int
    numneurons::Int
    W::Matrix{Float32}
    b::Vector{Float32}
    f::Function
    f′::Function
    function FullyConnected(inputsize, numneurons, transform::Function, derivative::Function)
        # Normal distributed random numbers with N(0, 0.1)
        weights = 1randn(numneurons, inputsize)
        bias = 1randn(numneurons)
        new(inputsize, numneurons, weights, bias, transform, derivative)
    end
end

function feed(layer::FullyConnected, a)
    layer.W * a .+ layer.b
end


function newsize(layer::FullyConnected, newsize)::FullyConnected
    FullyConnected(newsize, layer.numneurons, layer.f, layer.f′)
end

struct SoftMaxLayer <: Layer
    numneurons::Int
    #W::Matrix{Float32}
    f::Function
    function SoftMaxLayer(numneurons)
        new(numneurons, softmax)
    end
end

function feed(layer::SoftMaxLayer, a)
    a
end

struct NeuralNet
    layers::Vector{Layer}
    ∇cost::Function
    NeuralNet(layers::Vector{Layer}, ∇f::Function=MSE′) = new(layers, ∇f)
end
NeuralNet(layer::Layer) = NeuralNet([layer], MSE′)
NeuralNet() = NeuralNet(Layer[])

function NeuralNet(neurons::Vector{Int})
    net = NeuralNet()
    if length(neurons) > 0
        # First layer keeps the input 0 until an input is given
        push!(net.layers, FullyConnected(0, neurons[1], σ, ∇σ))
        for numneurons in neurons[2:end]
            addSigmoidLayer!(net, numneurons)
        end
    end
    net
end

function NeuralNet(architecture::AbstractArray{Tuple{Int, Symbol}})
    layers = Layer[]
    numneurons, activation = architecture[1]
    a, ∇a = getactivation(activation)
    push!(layers, FullyConnected(0, numneurons, a, ∇a))
    for i in 2:length(architecture)
        input, _ = architecture[i-1]
        numneurons, activation = architecture[i]
        a, ∇a = getactivation(activation)
        push!(layers, FullyConnected(input, numneurons, a, ∇a))
    end
    NeuralNet(layers)
end


function getactivation(f::Union{Symbol, Function})::Tuple{Function, Function}
    if f == σ || f == :sigmoid
        σ, ∇σ
    elseif f == relu || f == :relu
        relu, ∇relu
    elseif f == leakyrelu || f == :leakyrelu
        leakyrelu, ∇leakyrelu
    elseif f == :id
        x -> x, x -> 1
    else
        throw(KeyError("Activation $f not supported"))
    end
end

function addLayer!(net::NeuralNet, numneurons, transform::Function, derivative::Function)
    inputsize = try 
        net.layers[end].numneurons
    catch
        0
    end
    layer = FullyConnected(inputsize, numneurons, transform, derivative)
    push!(net.layers, layer)
end

function addSigmoidLayer!(net::NeuralNet, numneurons)
    addLayer!(net, numneurons, σ, ∇σ)
end

function addReluLayer!(net::NeuralNet, numneurons)
    addLayer!(net, numneurons, relu, ∇relu)
end

function addLeakyReluLayer!(net::NeuralNet, numneurons)
    addLayer!(net, numneurons, leakyrelu, ∇leakyrelu)
end

function addSoftmaxLayer!(net::NeuralNet)
    last = net.layers[end]
    push!(net.layers, SoftMaxLayer(last.numneurons))
end

function weave!(net, X, y)
    # X has dimension #predictors×#samples
    if length(net.layers) == 0
        push!(net.layers, FullyConnected(size(X, 2), 1, σ, ∇σ))
    else
        net.layers[1] = newsize(net.layers[1], size(X, 1))
    end
end

function backpropagation(net::NeuralNet, X, y)
    ∇b = [zeros(Float32, size(layer.b)) for layer in net.layers]
    ∇W = [zeros(Float32, size(layer.W)) for layer in net.layers]

    # Forward propagation
    a = X
    as = [a]
    zs = []
    for layer in net.layers
        z = layer.W * a .+ layer.b
        a = layer.f.(z)
        push!(zs, z)
        push!(as, a)
    end

    # Backpropagation
    # Handle final layer L
    ∇aᴸC = net.∇cost(as[end] |> softmax, y)
    δ = ∇aᴸC .* net.layers[end].f′.(zs[end])
    ∇b[end] .= δ
    ∇W[end] .= δ*as[end-1]'

    # Handle the remaining layers
    for l in 1:(length(net.layers)-1)
        ∂a∂z = net.layers[end-l].f′.(zs[end-l])
        δ = net.layers[end-l+1].W' * δ .* ∂a∂z
        ∇b[end-l] .= δ
        ∇W[end-l] .= δ * as[end-l-1]'
    end
    ∇b, ∇W
end


function feedforward(net::NeuralNet, X)
    for layer in net.layers
        X = layer.f.(layer.W*X .+ layer.b)
    end
    #X .= softmax(X)
    X
end

function evaluate(net::NeuralNet, X, labels)::Float32
    ŷ = predict(net, X)
    @assert size(ŷ) == size(labels)
    mean(labels .== ŷ)
end

function evaluate!(net::NeuralNet, o::Optimizer, X, labels)::Nothing
    ŷ = predict(net, X)
    @assert size(ŷ) == size(labels)
    o.loss[o.iterations] = mean(labels .== ŷ)

    !o.hasvalidationset && return

    ŷ = predict(net, o.validationset[1])
    o.validationloss[o.iterations] = mean(o.validationset[2]' .== ŷ)
    return
end

function logcrossentropy(net::NeuralNet, X, y)
    ŷ = feedforward(net, X)
    loss = logcrossentropy(y, ŷ)
end

function predict(net::NeuralNet, X)::Matrix{Int}
    probabilities = feedforward(net, X)
    return [imax[1] - 1 for imax in argmax(probabilities, dims=1)]
end

function MSE(net::NeuralNet, X, y)
    ŷ = feedforward(net, X)
    mean((y .- ŷ).^2)
end

function MSE′(ŷ, y)
    ŷ .- y
end


"""
    σ(x)

Sigmoid activation function
"""
σ(x::Real) = one(x)/(one(x) + exp(-x))
∇σ(x::Real) = σ(x)*(one(x) - σ(x))

"""
    relu(x)

Rectified Linear Unit activation function
"""
relu(x::Real) = max(zero(x), x)
∇relu(x::Real) = if x ≥ zero(x) one(x) else zero(x) end

"""
    leakyrelu(x)

Leaky Rectified Linear Unit activation function
"""
leakyrelu(x::Real, a = oftype(x, 0.01)) = max(a*x, x)
∇leakyrelu(x::Real, a = oftype(x, 0.01)) = if x ≥ zero(x) one(x) else a end


function softmax(xs::AbstractArray; dims=1)
    max_ = maximum(xs, dims=dims)
    exp_ = exp.(xs .- max_)
    exp_ ./ sum(exp_, dims=dims)
end

end
