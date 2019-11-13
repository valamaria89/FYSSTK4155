include("../src/classification.jl")
using Main.Classification
using Random
using PyCall
using Statistics
#using Gadfly
import PyPlot; const plt = PyPlot

Random.seed!(4)

sklsets = pyimport("sklearn.datasets")
observables, target = sklsets.make_classification(n_samples=2000, 
                                                  n_features=25,
                                                  n_informative=24,
                                                  n_redundant=1,
                                                  n_repeated=0,
                                                  n_classes=2,
                                                  random_state=42,
                                                  flip_y=0.05,
                                                  class_sep=1.0)

(Xtrain, Ytrain), (Xtest, Ytest) = traintestsplit(observables, target, 0.8, 2000)

Xtrain, Ytrain = (Xtrain, Ytrain) .|> collect ∘ transpose
Xtest, Ytest = (Xtest, Ytest) .|> collect ∘ transpose

Y = onehot(Ytrain)

gd_optim = GradientDescent(GDContext(learningrate=0.1, maxiterations=500))
optim = StochasticGradientDescent(SGDContext(learningrate=0.1, maxiterations=500))
optim2 = StochasticGradientDescent(SGDContext(learningrate=0.1, maxiterations=500))
addvalidationset!(optim2, Xtest, vec(Ytest))
addvalidationset!(gd_optim, Xtest, vec(Ytest))
addvalidationset!(optim, Xtest, vec(Ytest))
net = NeuralNet([(5, :relu), (2, :sigmoid)])
#addSoftmaxLayer!(net)
@time fit!(net, gd_optim, copy(Xtrain), copy(Y), copy(Ytrain))


net = NeuralNet([(5, :leakyrelu), (2, :sigmoid)])
@time fit!(net, optim2, copy(Xtrain), copy(Y), copy(Ytrain))

net = NeuralNet([(5, :relu), (2, :sigmoid)])
@time fit!(net, optim, copy(Xtrain), copy(Y), copy(Ytrain))

@show mean(predict(net, Xtrain) .== Ytrain)
@show mean(predict(net, Xtest) .== Ytest)

# plt = plot(x = 1:length(optim.loss), y = optim.loss,
     # Geom.line, Guide.xlabel("iterations"),
     # Guide.ylabel("loss"))
     #
fig, ax = plt.subplots()
line = ax.plot(gd_optim.loss, label="Gradient Descent")
ax.plot(gd_optim.validationloss, "--", c=line[1].get_color())
line = ax.plot(optim.loss, label="Stochastic")
ax.plot(optim.validationloss, "--", c=line[1].get_color())
line = ax.plot(optim2.loss, label="Stochastic Leaky")
ax.plot(optim2.validationloss, "--", c=line[1].get_color())
ax.set_ylabel("Accuracy")
ax.set_xlabel("Epochs")
ax.legend()
plt.show()
