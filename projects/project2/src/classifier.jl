module Classifiers
export LogisticClassifier, centerscale!, add_designmatrix!, fit!, designmatrix, predict, score

using ..Classification: Optimizer, GDContext, SGDContext, σ, accuracy
import ..Classification: fit!, addvalidationset!
#include("optimizer.jl")
# using .Optimizers: Optimizer
# import .Optimizers: fit!, addtracer!
# using .GD: GDContext, GradientDescent

# include("sgd.jl")
# using .SGD: SGDContext, StochasticGradientDescent

using Statistics

mutable struct LogisticClassifier
    optimizer::Optimizer
    designmatrix::Matrix{Float64}
    β::Vector{Float64}
    mean::Matrix{Float64}
    scale::Matrix{Float64}
    function LogisticClassifier(optimizer::T) where T<:Optimizer
        new(optimizer, Matrix{Float64}(undef, 0, 0),
            Float64[], Matrix{Float64}(undef, 0, 0),
            Matrix{Float64}(undef, 0, 0))
    end
end
function LogisticClassifier(context::GDContext)
    LogisticClassifier(GradientDescent(context))
end

function LogisticClassifier(context::SGDContext)
    LogisticClassifier(StochasticGradientDescent(context))
end

function centerscale!(classifier::LogisticClassifier)
    classifier.mean = mean(classifier.designmatrix, dims=1)
    classifier.scale = std(classifier.designmatrix, dims=1)
end

function designmatrix(classifier::LogisticClassifier, X::Matrix{Float64}; center=true,
                       scale=true, addintercept=true)::Matrix{Float64}
    mat = copy(X)
    center && mat .-= classifier.mean
    scale  && mat ./= classifier.scale
    if addintercept
        [ones(typeof(mat[1]), size(mat, 1)) mat]
    else
        mat
    end
end

function add_designmatrix!(classifier::LogisticClassifier, X::Matrix{Float64}; center=true,
                           scale=true, addintercept=true)
    classifier.designmatrix = copy(X)
    centerscale!(classifier)
    center && classifier.designmatrix .-= classifier.mean
    scale  && classifier.designmatrix ./= classifier.scale
    if addintercept
        classifier.designmatrix = [ones(typeof(classifier.designmatrix[1]), size(classifier.designmatrix, 1)) classifier.designmatrix]
    end
    classifier.β = zeros(Float64, size(classifier.designmatrix, 2))
end

function fit!(classifier::LogisticClassifier, X::Matrix{Float64}, y::Vector{T}) where T
    add_designmatrix!(classifier, X)
    fit!(classifier, y)
end

function fit!(classifier::LogisticClassifier, y::Vector{T}) where T
    fit!(classifier, convert(Vector{Float64}, y)::Vector{Float64})
end

function addvalidationset!(classifier::LogisticClassifier, X, y, earlystopping=false)
    X = designmatrix(classifier, X)
    addvalidationset!(classifier.optimizer, X, y, earlystopping)
end

function fit!(classifier::LogisticClassifier, y::Vector{Float64})
    @time fit!(classifier.optimizer,
               classifier.β,
               classifier.designmatrix,
               y)
end

function predict(classifier::LogisticClassifier, X)
    X = designmatrix(classifier, X)
    y = (σ.(X*classifier.β) .> 0.5) .|> Int
end

function addtracer!(classifier::LogisticClassifier, func)
    addtracer!(classifier.optimizer, func)
end

function score(classifier::LogisticClassifier, X, y)
    ŷ = predict(classifier, X)
    accuracy(ŷ, y)
end

end
