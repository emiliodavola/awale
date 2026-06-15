module Model

using Flux
using Serialization
using ..State: GameState, canonicalize

export create_model, predict, predict_batch, predict_raw, encode_state, save_model, load_model

mutable struct AwaleModel
    shared::Chain
    policy::Chain
    value::Chain
end

# Use @layer for Flux >= 0.15 to avoid deprecation warnings and ensure parameter tracking
Flux.@layer AwaleModel

function create_model()
    return AwaleModel(
        Chain(
            Dense(14 => 128, relu),
            Dense(128 => 128, relu)
        ),
        Chain(
            Dense(128 => 64, relu),
            Dense(64 => 6)
        ),
        Chain(
            Dense(128 => 64, relu),
            Dense(64 => 1, tanh)
        )
    )
end

function encode_state(s::GameState)::Vector{Float32}
    # Board is NTuple{12}, captured is NTuple{2} -> Total 14
    board = Float32.(collect(s.board)) ./ 48f0
    captured = Float32[
        s.captured[1] / 48f0,
        s.captured[2] / 48f0
    ]
    return vcat(board, captured)
end

function predict_raw(model::AwaleModel, X::AbstractMatrix{Float32})
    shared_out = model.shared(X)
    logits = model.policy(shared_out)
    value  = model.value(shared_out)
    return logits, value
end

function predict(model::AwaleModel, s::GameState)
    s_can = canonicalize(s)
    x = reshape(encode_state(s_can), :, 1)
    logits, value = predict_raw(model, x)
    return vec(logits), value[1]
end

function predict_batch(model::AwaleModel, states::Vector{GameState})
    X = hcat([encode_state(canonicalize(s)) for s in states]...)
    return predict_raw(model, X)
end

function save_model(model::AwaleModel, path::AbstractString)
    open(path, "w") do io
        Serialization.serialize(io, model)
    end
    return path
end

function load_model(path::AbstractString)::AwaleModel
    open(path, "r") do io
        return Serialization.deserialize(io)
    end
end

end # module
