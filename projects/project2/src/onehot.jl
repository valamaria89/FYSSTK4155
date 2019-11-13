
function onehot(l, labels)
    i = something(findfirst(isequal(l), labels), 0)
    i > 0 || error("Value $l is not in labels")
    x = zeros(Int8, length(labels))'
    x[i] = 1
    x
end

function onehot(l::AbstractArray, labels)
    m = Matrix{Int8}(undef, length(labels), length(l))
    for (i, elem) in enumerate(l)
        m[:, i] = onehot(elem, labels)
    end
    m
end

function onehot(l)
    u = unique(l)
    onehot(l, minimum(u):maximum(u))
end

function onecold(l::AbstractMatrix, labels)
    m = Matrix{eltype(labels)}(undef, 1, size(l, 2))
    for (i, col) in enumerate(eachcol(l))
        m[1, i] = labels[argmax(col)]
    end
    m
end
