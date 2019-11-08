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
    loss::Vector{Float64}
    validationset::Tuple{Matrix{Float64}, Vector{Float64}}
    validationloss::Vector{Float64}
end

function GradientDescent(context=GDContext())
    GradientDescent(context, false, false, false, 0, Float64[], 
                              (zeros(Float64, 1, 1), Float64[]), Float64[])
end

function fit!(optim::GradientDescent, β::Vector{Float64},
              X::Matrix{Float64}, y::Vector{Float64})
    context = optim.context
    previousloss = -Inf
    optim.loss = zeros(Float64, context.maxiterations)
    # Preallocate everything
    ŷ         = zeros(Float64, length(y))
    residuals = zeros(Float64, length(y))
    ∇array    = zeros(Float64, (1, length(β)))
    ∇         = @view ∇array[1, :]
    tmp1      = zeros(Float64, length(y))
    tmp       = zeros(Float64, size(X))
    loss      = [0.0]
    tmp2      = zeros(Float64, length(y))

    @inbounds for i in 1:context.maxiterations
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
