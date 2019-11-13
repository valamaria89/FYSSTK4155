module Optimizers
using Statistics
export Optimizer, OptimizerContext, accuracy, logcrossentropy, fit!, σ, mse, logcrossentropy!, sigmoid!, stopearly!, addvalidationset!, postprocess!

"""
    Optimizer

Abstract type for all optimizers
"""
abstract type Optimizer end
abstract type OptimizerContext end

function fit!(optim::Optimizer, β, X, y)
    @error "Not implemented"
end

function addvalidationset!(optim::Optimizer, X, y, earlystopping=false)
    optim.validationset = (X, y)
    optim.validationloss = zeros(Float32, optim.context.maxiterations)
    optim.earlystopping = earlystopping
    optim.hasvalidationset = true
end

function stopearly!(optim::Optimizer, β, step::Int)::Bool
    if !optim.hasvalidationset
        return false
    end

    history = 3
    yhat = σ.(optim.validationset[1]*β)
    optim.validationloss[step] = logcrossentropy(optim.validationset[2], 
                                                 yhat)
    if step ≤ history || !optim.earlystopping
        return false
    end
    limit = mean(optim.validationloss[(step - history):(step - 1)])
    if optim.validationloss[step] > limit
        return true
    else
        return false
    end
end

function postprocess!(optim::Optimizer)::Nothing
    resize!(optim.loss, optim.iterations)
    if optim.hasvalidationset
        if optim.converged
            resize!(optim.validationloss, optim.iterations-1)
        else
            resize!(optim.validationloss, optim.iterations)
        end
    end
    return
end


function accuracy(observations, predictions)
    truepositive = observations .== Int.(predictions .> 0.5)
    sum(truepositive)/length(observations)
end

function lognull(x::T)::T where T<:Real
    x == zero(x) && return zero(x)
    log(x)
end

function logcrossentropy(observations, predictions)
    -mean(@. observations * lognull(predictions) + (1-observations) * lognull(1-predictions))
end

function logcrossentropy!(output, tmp, observations, predictions)
    @. tmp = - observations * lognull(predictions) - (1-observations) * lognull(1-predictions)
    mean!(output, tmp)
end

function mse(observations, predictions)
    mean((observations .- predictions).^2)
end

sigmoid(x) = 1 /(1 + exp(-x))
σ = sigmoid
function sigmoid!(y, x)
    for i in eachindex(y)
        y[i] = 1/(1 + exp(-x[i]))
    end
end

end
