include("../src/classification.jl")
using Main.Classification
using Random
using PyCall
using Statistics
import PyPlot; const plt = PyPlot

metrics = pyimport("sklearn.metrics");
Random.seed!(5);
sklsets = pyimport("sklearn.datasets")
observables, target = sklsets.make_classification(n_samples=5000, n_features=20, n_informative=15,
n_redundant=1, n_repeated=0, n_classes=2, random_state=1, flip_y=0.05, class_sep=1.0, shuffle=true)
(Xtrain, Ytrain), (Xtest, Ytest) = traintestsplit(observables, target, 0.8, 5000);

sgd = SGDContext(learningrate=0.1, tolerance=1e-7, maxiterations=1000)
clf = LogisticClassifier(StochasticGradientDescent(sgd))
add_designmatrix!(clf, Xtrain)
addvalidationset!(clf, Xtest, Ytest, false)
@time fit!(clf, Ytrain)
@show clf.optimizer.converged
@show mean(Ytest .== predict(clf, Xtest))

fig, ax = plt.subplots()
ax.plot(clf.optimizer.loss)
ax.plot(clf.optimizer.validationloss)
plt.show()

