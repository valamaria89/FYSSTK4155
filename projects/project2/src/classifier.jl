module Classifiers
export LogisticClassifier, centerscale!, add_designmatrix!, fit!, designmatrix, predict, score, traintestsplit

using ..Classification: Optimizer, GDContext, SGDContext, σ, accuracy
import ..Classification: fit!, addvalidationset!
using Random: shuffle
using Statistics

mutable struct LogisticClassifier
    optimizer::Optimizer
    designmatrix::Matrix{<:Real}
    β::Vector{Float32}
    mean::Matrix{<:Real}
    scale::Matrix{<:Real}
    function LogisticClassifier(optimizer::V) where V<:Optimizer
        new(optimizer, Matrix{Float32}(undef, 0, 0),
            Float64[], Matrix{Float32}(undef, 0, 0),
            Matrix{Float32}(undef, 0, 0))
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

function add_designmatrix!(classifier::LogisticClassifier, X::Matrix{T}; center=true,
                           scale=true, addintercept=true) where T<:Real
    classifier.designmatrix = copy(X)
    centerscale!(classifier)
    center && classifier.designmatrix .-= classifier.mean
    scale  && classifier.designmatrix ./= classifier.scale
    if addintercept
        classifier.designmatrix = [ones(typeof(classifier.designmatrix[1]), size(classifier.designmatrix, 1)) classifier.designmatrix]::Matrix{T}
    end
    classifier.β = zeros(Float64, size(classifier.designmatrix, 2))
end

function fit!(classifier::LogisticClassifier, X::Matrix{<:Real}, y::Vector{<:Real})
    add_designmatrix!(classifier, X)
    fit!(classifier, y)
end

function addvalidationset!(classifier::LogisticClassifier, X, y, earlystopping=false)
    X = designmatrix(classifier, X)
    addvalidationset!(classifier.optimizer, X, y, earlystopping)
end

function fit!(classifier::LogisticClassifier, y::Vector{<:Real})
    fit!(classifier.optimizer,
         classifier.β,
         classifier.designmatrix,
         y)
end

function predict(classifier::LogisticClassifier, X)
    X = designmatrix(classifier, X)
    y = (σ.(X*classifier.β) .> 0.5) .|> Int
end

function score(classifier::LogisticClassifier, X, y)
    ŷ = predict(classifier, X)
    accuracy(ŷ, y)
end

function traintestsplit(X, y, splitratio, portion=nothing)
    if portion === nothing
        portion = size(X, 1)
    end
    if portion > size(X, 1)
        @warn "Data set too small; using all"
        portion = size(X, 1)
    end
    M = shuffle(1:length(y))
    traintest = splitratio*portion |> x -> round(Int, x)
    Xtrain = X[M, :][1:traintest, :]
    Ytrain = y[M][1:traintest]
    Xtest  = X[M, :][traintest+1:portion, :]
    Ytest  = y[M][traintest+1:portion];
    (Xtrain, Ytrain), (Xtest, Ytest)
end

end
