using Test
include("../src/classification.jl")
using Main.Classification


@testset "One hot encoding" begin
    @test onehot(0, [0, 1]) == Int8[1 0]
    @test onehot(7, 5:8) == Int8[0 0 1 0]
    @test onehot(:a, [:c, :g, :a, :5]) == Int8[0 0 1 0]
    @test onehot([2 3 4 2 2], 1:4) == Int8[0 0 0 0 0; 1 0 0 1 1; 0 1 0 0 0; 0 0 1 0 0]
    @test onehot([2 3 4 2 1]) == Int8[0 0 0 0 1; 1 0 0 1 0; 0 1 0 0 0; 0 0 1 0 0]
end

@testset "One cold decoding" begin
    @test onecold(onehot([2 3 4 2 1]), 1:4) == [2 3 4 2 1]
end
