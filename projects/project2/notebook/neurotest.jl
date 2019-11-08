include("../src/classification.jl")
using Main.Classification
using Random
using PyCall
using Statistics
#using Gadfly
import PyPlot; const plt = PyPlot

Random.seed!(4)


function traintestsplit(X, y, splitratio, portion)
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

sklsets = pyimport("sklearn.datasets")
observables, target = sklsets.make_classification(n_samples=2000, n_features=25, n_informative=24,
n_redundant=1, n_repeated=0, n_classes=2, random_state=42, flip_y=0.05, class_sep=1.0)
(Xtrain, Ytrain), (Xtest, Ytest) = traintestsplit(observables, target, 0.8, 2000)
Xtrain, Ytrain = (Xtrain, Ytrain) .|> collect ∘ transpose
@show Xtrain |> size
@show Ytrain |> size

net = NeuralNet([8, 3, 2])
@show map(x -> size(x.W), net.layers)
gd = GDContext(learningrate=0.1, maxiterations=100)
optim = @time fit!(net, gd, Xtrain, Ytrain)


# plt = plot(x = 1:length(optim.loss), y = optim.loss,
     # Geom.line, Guide.xlabel("iterations"),
     # Guide.ylabel("loss"))
fig = plt.plot(optim.loss)
display(fig)
plt.show()
