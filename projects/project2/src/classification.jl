module Classification
export Optimizer, OptimizerContext, GDContext, GradientDescent,
       StochasticGradientDescent, SGDContext, LogisticClassifier,
       fit!, add_designmatrix!, designmatrix, predict, σ, logcrossentropy,
       mse, predict, sigmoid!, logcrossentropy!, NesterovGradientDescent,
       NAGDContext, partition, addvalidationset!, stopearly!, postprocess!,
       score, accuracy, NeuralNet, weave!, addLayer!, addSigmoidLayer!, 
       sigmoidactivation!, FullyConnected, forwardpropagation, backpropagation,
       feedforward, minibatch, traintestsplit, onehot, onecold, evaluate,
       addSoftmaxLayer!

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
using .Classifiers: LogisticClassifier, designmatrix, add_designmatrix!,  predict, score, traintestsplit
import .Classifiers: fit!
include("neuralnet.jl")
using .NeuralNetwork

include("gd_net.jl")
include("sgd_net.jl")
include("onehot.jl")
end
