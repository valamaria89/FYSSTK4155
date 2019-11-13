module GD
export GDContext, GradientDescent, fit!

using ..Classification: OptimizerContext, Optimizer, σ, logcrossentropy, stopearly!, sigmoid!, logcrossentropy!, postprocess!
import ..Classification: fit!
using LinearAlgebra: mul!
using Statistics: mean!

struct GDContext <: OptimizerContext
    learningrate::Float64
    tolerance::Float64
    maxiterations::Int
end

GDContext(;learningrate=0.1, tolerance=eps(Float64),
          maxiterations=1000) = GDContext(learningrate, tolerance, maxiterations)

mutable struct GradientDescent <: Optimizer
    context::GDContext
    converged::Bool
    earlystopping::Bool
    hasvalidationset::Bool
    iterations::Int
    loss::Vector{Float32}
    # Lower memory usage is better than high precision
    validationset::Tuple{Matrix{Float32}, Vector{Int8}}
    validationloss::Vector{Float32}
end

function GradientDescent(context=GDContext())
    GradientDescent(context, false, false, false, 0, Float32[], 
                   (zeros(Float32, 1, 1), Int8[]), Float32[])
end

function fit!(optim::GradientDescent, β::Vector{T},
              X::Matrix{<:Real}, y::Vector{<:Real}) where {T<:Real}
    context = optim.context
    previousloss = -Inf
    optim.loss = zeros(Float32, context.maxiterations)
    # Preallocate everything
    ŷ         = zeros(Float32, length(y))
    residuals = zeros(Float32, length(y))
    ∇array    = zeros(T, (1, length(β)))
    ∇         = @view ∇array[1, :]
    tmp1      = zeros(T, length(y))
    tmp       = zeros(T, size(X))
    loss      = [0.0]
    tmp2      = zeros(Float32, length(y))

    for i in 1:context.maxiterations
        optim.iterations += 1
        # ŷ = σ(X*β)
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

        # Gradient descent
        residuals .= ŷ .- y
        tmp .= X.*residuals
        mean!(∇array, tmp)

        β .-= context.learningrate .* ∇

        if stopearly!(optim, β, i)
            break
        end
    end
    postprocess!(optim)
end

end
