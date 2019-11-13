module SGD
export SGDContext, StochasticGradientDescent, fit!, partition
using ..Classification: OptimizerContext, Optimizer, σ, logcrossentropy, sigmoid!, logcrossentropy!, stopearly!, postprocess!
import ..Classification: fit!
using LinearAlgebra: mul!
#using Random: shuffle!
using Statistics: mean!

struct SGDContext <: OptimizerContext
    learningrate::Float32
    decayrate::Float32
    tolerance::Float32
    maxiterations::Int
    batchsize::Int
end
SGDContext(;learningrate=0.1, decayrate=0.8, tolerance=eps(Float32),
           maxiterations=1000, batchsize=100) = SGDContext(learningrate, decayrate,
                                                         tolerance, maxiterations, batchsize)

mutable struct StochasticGradientDescent <: Optimizer
    context::SGDContext
    converged::Bool
    earlystopping::Bool
    hasvalidationset::Bool
    iterations::Int
    loss::Vector{Float32}
    validationset::Tuple{Matrix{Float32}, Vector{Int8}}
    validationloss::Vector{Float32}
end

function StochasticGradientDescent(context=SGDContext())
    StochasticGradientDescent(context, false, false, false, 0, Float32[], 
                              (zeros(Float32, 1, 1), Int8[]), Float32[])
end

function fit!(optim::StochasticGradientDescent, β::Vector{T}, X, y) where {T<:Real}
    context = optim.context
    previousloss = -Inf
    optim.loss = zeros(Float32, context.maxiterations)
    momentum = zeros(Float32, size(β))
    n, k = size(X)
    m = context.batchsize
    #shuffled  = collect(1:n)
    ŷbatch    = zeros(Float32, m)
    ŷ         = zeros(Float32, n)
    residuals = zeros(Float32, m)
    ∇array    = zeros(T, (1, length(β)))
    ∇         = @view ∇array[1, :]
    tmp1batch = zeros(Float32, m)
    tmp1      = zeros(T, n)
    tmp2      = zeros(Float32, n)
    tmp       = zeros(T, (m, k))
    loss      = [0.0]
    Xbatch    = zeros(Float32, (m, k))
    ybatch    = zeros(Float32, m)
    batches   = partition(n, context.batchsize)

    for i in 1:context.maxiterations
        optim.iterations += 1
        shuffle!(X, y)

        for batch in batches
            move!(X, y, Xbatch, ybatch, batch)

            mul!(tmp1batch, Xbatch, β)
            sigmoid!(ŷbatch, tmp1batch)

            # Gradient descent
            residuals .= ŷbatch .- ybatch
            tmp       .= Xbatch.*residuals
            mean!(∇array, tmp)
            @. momentum  = context.decayrate*momentum + context.learningrate * ∇
            β          .-= momentum
        end

        mul!(tmp1, X, β)
        sigmoid!(ŷ, tmp1)

        logcrossentropy!(loss, tmp2, y, ŷ)
        optim.loss[i] = loss[1]
        if abs(previousloss - loss[1]) < context.tolerance
            optim.converged = true
            break
        else
            previousloss = loss[1]
        end

        if stopearly!(optim, β, i)
            break
        end
    end
    postprocess!(optim)
end

function partition(length::Number, batchsize::Number)::Vector{UnitRange{Int}}
    slices = Vector{UnitRange{Int}}(undef, length ÷ batchsize)
    lastaddon = length % batchsize
    start = 1
    i = 1
    while true
        stop = start + batchsize - 1
        if stop > length
            break
        end
        slices[i] = start:stop
        start = stop + 1
        i += 1
    end
    #slices[end] = first(slices[end]):(last(slices[end])+lastaddon)
    # I don't know how to do the last batch
    pop!(slices)
    slices
end

function shuffle!(X::Matrix{T} where T, y::Vector{V} where V)
    n = length(y)
    #@assert n == size(X, 1)
    #@assert n > 1
    j = 0
    @inbounds for i in n:-1:2
        @fastmath j = 1 + abs(rand(Int))%n
        swaprows!(X, i, j)
        y[i], y[j] = y[j], y[i]
    end
end

function swaprows!(X, i, j)
    for k in 1:size(X, 2)
        X[i, k], X[j, k] = X[j, k], X[i, k]
    end
end

function move!(X, y, X̃, ỹ, indices)
    j = 1
    for i in indices
        moverow!(X, X̃, i, j)
        ỹ[j] = y[i]
        j += 1
    end
end

function moverow!(X, X̃, i, j)
    for k in 1:size(X, 2)
        X̃[j, k] = X[i, k]
    end
end

end
