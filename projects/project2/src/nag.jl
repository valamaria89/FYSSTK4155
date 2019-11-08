
module NAGD
export NAGDContext, NesterovGradientDescent, fit!
using ..Classification: OptimizerContext, Optimizer, σ, logcrossentropy, sigmoid!, logcrossentropy!, partition, stopearly!, postprocess!
import ..Classification: fit!
using LinearAlgebra: mul!
#using Random: shuffle!
using Statistics: mean!

struct NAGDContext <: OptimizerContext
    learningrate::Float64
    decayrate::Float64
    tolerance::Float64
    maxiterations::Int
    batchsize::Int
end
NAGDContext(;learningrate=0.1, decayrate=0.8, tolerance=eps(Float64),
           maxiterations=1000, batchsize=8) = NAGDContext(learningrate, decayrate, tolerance, maxiterations, batchsize)

mutable struct NesterovGradientDescent <: Optimizer
    context::NAGDContext
    converged::Bool
    earlystopping::Bool
    hasvalidationset::Bool
    iterations::Int
    loss::Vector{Float64}
    validationloss::Vector{Float64}
    validationset::Tuple{Matrix{Float64}, Vector{Float64}}
end

function NesterovGradientDescent(context=NAGDContext())
    NesterovGradientDescent(context, false, false, false, 0, Float64[], Float64[], (zeros(Float64, (1, 1)), Float64[]))
end

function fit!(optim::NesterovGradientDescent, β, X, y)
    context = optim.context
    previousloss = -Inf
    optim.loss = zeros(Float64, context.maxiterations)
    momentum = zeros(Float64, size(β))
    n, k = size(X)
    m = context.batchsize
    βnear     = zeros(Float64, length(β))
    ŷbatch    = zeros(Float64, m)
    ŷ         = zeros(Float64, n)
    residuals = zeros(Float64, m)
    ∇array    = zeros(Float64, (1, length(β)))
    ∇         = @view ∇array[1, :]
    tmp1batch = zeros(Float64, m)
    tmp1      = zeros(Float64, n)
    tmp2      = zeros(Float64, n)
    tmp       = zeros(Float64, (m, k))
    loss      = [0.0]
    Xbatch    = zeros(Float64, (m, k))
    ybatch    = zeros(Float64, m)
    batches   = partition(n, context.batchsize)
    batch     = 1:1

    @inbounds for i in 1:context.maxiterations
        optim.iterations += 1
        shuffle!(X, y)

        for batch in batches
            move!(X, y, Xbatch, ybatch, batch)

            @. βnear = β + context.decayrate * momentum
            mul!(tmp1batch, Xbatch, βnear)
            sigmoid!(ŷbatch, tmp1batch)

            # Gradient descent
            residuals .= ŷbatch .- ybatch
            tmp       .= Xbatch.*residuals
            mean!(∇array, tmp)

            @. momentum = context.decayrate*momentum - context.learningrate * ∇
            β         .+= momentum
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
