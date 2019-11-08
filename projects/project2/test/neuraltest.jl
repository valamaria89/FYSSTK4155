using Test
include("../src/classification.jl")
using Main.Classification


@testset "Construct NeuralNet" begin
    X = [1 2 3 4 5
         1 2 3 4 5
         0 1 1 2 5]
    y = [1 1 0 1 1]
    # @testset "Empty" begin
        # net = NeuralNet()
        # @test length(net.layers) == 0
        # weave!(net, X, y)
        # @test length(net.layers) == 1
        # @test net.layers[1].inputsize == size(X, 2)
        # @test net.layers[1].numneurons == 1
    # end
    # @testset "Empty 2" begin
        # net = NeuralNet([])
        # @test length(net.layers) == 0
        # weave!(net, X, y)
        # @test length(net.layers) == 1
        # @test net.layers[1].inputsize == size(X, 2)
        # @test net.layers[1].numneurons == 1
    # end
    # @testset "Single" begin
        # net = NeuralNet([5])
        # @test length(net.layers) == 1
        # weave!(net, X, y)
        # @test length(net.layers) == 2
        # @test net.layers[1].inputsize == size(X, 2)
        # @test net.layers[1].numneurons == 5
        # @test net.layers[2].inputsize == 5
        # @test net.layers[2].numneurons == 1
    # end
    #
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
    # @testset "Build" begin
        # net = NeuralNet([5, 6])
        # @test length(net.layers) == 2
        # addSigmoidLayer!(net, 10)
        # addSigmoidLayer!(net, 20)
        # @test length(net.layers) == 4
        # @test net.layers[3].numneurons == 10
        # @test net.layers[3].inputsize == 6
        # @test net.layers[4].inputsize == 10
        # weave!(net, X, y)
        # @test length(net.layers) == 5
        # @test net.layers[1].inputsize == size(X, 2)
        # @test size(net.layers[1].weights, 1) == 5
        # @test size(net.layers[1].weights, 2) == size(X, 2) + 1
        # @test net.layers[1].numneurons == 5
        # @test net.layers[end].inputsize == 20
        # @test net.layers[end].numneurons == 1
    # end
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
    # @testset "Forward Propogation" begin
        # net = NeuralNet([8, 6])
        # weave!(net, X, y)
        # res = forwardpropagation(net, X)
        # @test size(res) == (3, 1)
        # @test res[1] ≈ res[2]
    # end
    @testset "Backward Propogation" begin
        net = NeuralNet([8, 6, 1])
        weave!(net, X, y)
        #forwardpropagation(net, X)
        backpropagation(net, X, y)
        @test true
    end
    # @testset "Minibatch iteration" begin
        # net = NeuralNet([8, 6, 1])
        # weave!(net, X, y)
        # xbatch = [[1 1
                   # 0 0
                   # 1 2],
                  # [1 1
                   # 1 0
                   # 1 1]]

        # ybatch = [[0 1], [0 1]]
        # minibatch(net, collect(zip(xbatch, ybatch)))
        # @test true
    # end
    @testset "Gradient Descent" begin
        net = NeuralNet([8, 6, 1])
        #forwardpropagation(net, X)
        gd = GradientDescent()
        fit!(net, gd, X, y)
        @test true
    end
end
