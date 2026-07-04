module Model

using TOML
using Flux
using Serialization
using ..State: GameState, canonicalize, encode_state
using ..Utils: fnv1a64

export create_model, predict, predict_batch, predict_raw, encode_state, save_model, load_model

mutable struct AwaleModel
    shared::Chain
    policy::Chain
    value::Chain
end

# Use @layer for Flux >= 0.15 to avoid deprecation warnings and ensure parameter tracking
Flux.@layer AwaleModel

function create_model(config_path::String="src/Awale/config.toml")
    config = TOML.parsefile(config_path)
    model_cfg = config["model"]

    act_map = Dict(
        "relu" => relu,
        "tanh" => tanh,
        "identity" => identity,
    )

    shared_layers = [Dense(layer["in"] => layer["out"], act_map[layer["activation"]]) for layer in model_cfg["layers"]["shared"]]
    policy_layers = [Dense(layer["in"] => layer["out"], act_map[layer["activation"]]) for layer in model_cfg["layers"]["policy"]]
    value_layers = [Dense(layer["in"] => layer["out"], act_map[layer["activation"]]) for layer in model_cfg["layers"]["value"]]

    return AwaleModel(Chain(shared_layers...), Chain(policy_layers...), Chain(value_layers...))
end

function predict_raw(model::AwaleModel, X::AbstractMatrix{Float32})
    shared_out = model.shared(X)
    logits = model.policy(shared_out)
    value = model.value(shared_out)
    return logits, value
end

function predict(model::AwaleModel, s::GameState)
    s_can = canonicalize(s)
    x = reshape(vec(encode_state(s_can)), :, 1)
    logits, value = predict_raw(model, x)
    return vec(logits), value[1]
end

function predict_batch(model::AwaleModel, states::Vector{GameState})
    X = hcat([vec(encode_state(canonicalize(s))) for s in states]...)
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
