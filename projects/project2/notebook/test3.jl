include("../src/classification.jl")
using Main.Classification
using Random
using PyCall
using Statistics

metrics = pyimport("sklearn.metrics");
Random.seed!(5);
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
datasets = pyimport("sklearn.datasets")
cancer = datasets.load_breast_cancer()
wine = datasets.load_wine()
set = cancer
(Xtrain, Ytrain), (Xtest, Ytest) = traintestsplit(set["data"], set["target"], 0.8, 1000)

nagd = NAGDContext(learningrate=0.1, tolerance=1e-7, maxiterations=100000)
clf = LogisticClassifier(NesterovGradientDescent(nagd))
add_designmatrix!(clf, Xtrain)
addvalidationset!(clf, Xtest, Ytest)
@time fit!(clf, Ytrain)
@show clf.optimizer.converged
@show mean(Ytest .== predict(clf, Xtest))
