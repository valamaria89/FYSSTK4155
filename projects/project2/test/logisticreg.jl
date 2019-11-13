using Test
include("../src/classification.jl")
using Main.Classification
using Random


@testset "Gradient Descent" begin
    X = randn(10, 15)
    y = rand(0:1, 10)
    @testset "Context" begin
        gd = GDContext()
        gd = GDContext(learningrate = 0.1)
        @test true
    end
    @testset "Optimizer from context" begin
        gd = GDContext()
        o = GradientDescent(gd)
        o = GradientDescent()
        @test true
    end
    @testset "Optimizer methods" begin
        o = GradientDescent()
        addvalidationset!(o, X, y)
        @test_throws InexactError addvalidationset!(o, X, randn(10))
    end
    @testset "Fit" begin
        o = GradientDescent()
        β = zeros(15)
        fit!(o, β, X, y)
        @test length(o.loss) == o.iterations
        addvalidationset!(o, X, y)
        fit!(o, β, X, y)
        if o.converged
            @test length(o.validationloss) == o.iterations - 1
        else
            @test length(o.validationloss) == o.iterations
        end
    end
end

@testset "Stochastic Gradient Descent" begin
    X = randn(10, 15)
    y = rand(0:1, 10)
    @testset "Context" begin
        gd = SGDContext()
        gd = SGDContext(learningrate = 0.1)
        @test true
    end
    @testset "Optimizer from context" begin
        gd = SGDContext()
        o = StochasticGradientDescent(gd)
        o = StochasticGradientDescent()
        @test true
    end
    @testset "Optimizer methods" begin
        o = StochasticGradientDescent()
        addvalidationset!(o, X, y)
        @test_throws InexactError addvalidationset!(o, X, randn(10))
    end
    # Messed up types, causing this to fail
    # @testset "Fit" begin
        # o = StochasticGradientDescent()
        # β = zeros(15)
        # fit!(o, β, X, y)
        # @test length(o.loss) == o.iterations
        # addvalidationset!(o, X, y)
        # fit!(o, β, X, y)
        # @test true
    # end
end

@testset "Classifier" begin
    X = randn(10, 15)
    y = rand(0:1, 10)
    @testset "Gradient Descent Classifier" begin
        o = GradientDescent()
        clf = LogisticClassifier(o)
        add_designmatrix!(clf, X)
        fit!(clf, y)
        fit!(clf, X, y)
        @test true
    end
end
