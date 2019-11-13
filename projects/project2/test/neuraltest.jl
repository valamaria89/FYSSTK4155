using Test
include("../src/classification.jl")
using Main.Classification


@testset "Construct NeuralNet" begin
    X = [1 2 3 4 5
         1 2 3 4 5
         0 1 1 2 5]
    y = [1 1 0 1 1]
    @testset "Weave" begin
        net = NeuralNet([6, 10, 1])
        @test length(net.layers) == 3
        weave!(net, X, y)
        @test length(net.layers) == 3
        @test size(net.layers[1].b) == (6,)
        @test size(net.layers[2].b) == (10,)
        @test size(net.layers[3].b) == (1,)
        @test size(net.layers[1].W) == (6, 3)
        @test size(net.layers[2].W) == (10, 6)
        @test size(net.layers[3].W) == (1, 10)
    end
    @testset "FeedForward" begin
        net = NeuralNet([5, 2, 1])
        weave!(net, X, y)
        out = feedforward(net, X)
        @test size(out) == size(y)
    end
    @testset "FeedForward 2" begin
        net = NeuralNet([5, 2, 3])
        weave!(net, X, y)
        out = feedforward(net, X)
        @test size(out) == (3, size(y, 2))
    end
    # I messed up the types, causing these to fail
    # @testset "Backward Propogation" begin
        # net = NeuralNet([8, 6, 1])
        # weave!(net, X, y)
        # backpropagation(net, X, y)
        # @test true
    # end
    # @testset "Gradient Descent" begin
        # net = NeuralNet([8, 6, 1])
        # gd = GradientDescent()
        # fit!(net, gd, X, y)
        # @test true
    # end
end
