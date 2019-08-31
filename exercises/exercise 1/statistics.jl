module Duck
using LinearAlgebra
using FreqTables
export leastsquares

function leastsquares(x, y, Ω; order::T = 1) where T<:Integer
    X = ones((length(x), order))
    for n in 1:order
        X[:, n+1] = x.^n
    end

    if ndims(Ω) == 1
        Ω = diagm(0 => Ω)
    end

    w = inv(Ω)  # Weights defined as 1/σ²
    Xᵀ = transpose(X)
    kω = inv(Xᵀ*w*X)*Xᵀ*w
	coeffs = kω*y
    Σ = kω*Ω*kω'
    coeffs, Σ
end

function leastsquares(x, y; order::T = 1) where T<:Integer
    X = ones((length(x), order+1))
    for n in 1:order
        X[:, n+1] = x.^n
    end
    inv(transpose(X)*X)*transpose(X)*y
end

mutable struct RNG
    #= Simple random number generator =#
    value::Float64
end

function Base.iterate(rng::RNG, state::Int64)
    #= Park-Miller-Carta Pseudo-Random Number Generator =#
    a = 16807
    m = 2147483647
    q = 127773
    r = 2836
    MASK = 123459876
    am = 1/m
    state = state ⊻ MASK
    k::Int64 = round(state/q)
    state = a*(state - k*q) - r*k
    state += if state < 0 m else 0 end
    rng.value = am*state
    state = state ⊻ MASK
    (rng.value, state)
end

function rngarray(N)
    A = zeros(N)
    i = 1
    @inbounds for random in RNG(abs(rand(Int)) % 10492841)
        A[i] = random
        i += 1
        if i ≥ N+1
            break
        end
    end
    A
end

function israndom(randomnums::Array{Int64, 1}, r)
    numsamples = length(randomnums)
    if numsamples ≤ 10r
        return false
    end

    nr = numsamples/r
    # Frequency of randoms
    ht = freqtable(randomnums).array
    χ² = sum((v - nr)^2 for v in ht)/nr
    return abs(χ² - r) / 2*√r
end

end
