module Classification
export Optimizer, OptimizerContext, GDContext, GradientDescent, StochasticGradientDescent, SGDContext, LogisticClassifier, fit!, add_designmatrix!, designmatrix, predict, σ, logcrossentropy, mse, predict, sigmoid!, logcrossentropy!, NesterovGradientDescent, NAGDContext, partition, addvalidationset!, stopearly!, postprocess!, score, accuracy, NeuralNet, weave!, addLayer!, addSigmoidLayer!, sigmoidactivation!, FullyConnected, forwardpropagation, backpropagation, feedforward, minibatch
include("optimizer.jl")
using .Optimizers
include("gd.jl")
import .GD: GDContext, GradientDescent, fit!
include("sgd.jl")
import .SGD: SGDContext, StochasticGradientDescent, fit!
using .SGD: partition
include("nag.jl")
import .NAGD: NAGDContext, NesterovGradientDescent, fit!
include("classifier.jl")
using .Classifiers: LogisticClassifier, designmatrix, add_designmatrix!,  predict, score
import .Classifiers: fit!
include("neuralnet.jl")
using .NeuralNetwork
# using Reexport
# @reexport using .Optimizers
# @reexport using .GD
# @reexport using .SGD
# @reexport using .Classifiers
end
