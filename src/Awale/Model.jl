module Model

using TOML
using Flux
using Serialization
using ..State: GameState, canonicalize, encode_state
using ..Utils: fnv1a64

export create_model, predict, predict_batch, predict_raw, encode_state, save_model, load_model

const DEFAULT_MODEL_ARCHITECTURE = "mlp"

mutable struct AwaleModel
    shared::Chain
    policy::Chain
    value::Chain
end

# Use @layer for Flux >= 0.15 to avoid deprecation warnings and ensure parameter tracking
Flux.@layer AwaleModel

function model_architecture(model_cfg)::String
    architecture = get(model_cfg, "architecture", DEFAULT_MODEL_ARCHITECTURE)
    return lowercase(String(architecture))
end

function build_mlp_model(model_cfg)
    act_map = Dict(
        "relu" => relu,
        "tanh" => tanh,
        "identity" => identity,
    )

    haskey(model_cfg, "layers") || throw(ArgumentError("Model configuration is missing the 'layers' section."))
    shared_layers = [Dense(layer["in"] => layer["out"], act_map[layer["activation"]]) for layer in model_cfg["layers"]["shared"]]
    policy_layers = [Dense(layer["in"] => layer["out"], act_map[layer["activation"]]) for layer in model_cfg["layers"]["policy"]]
    value_layers = [Dense(layer["in"] => layer["out"], act_map[layer["activation"]]) for layer in model_cfg["layers"]["value"]]

    return AwaleModel(Chain(shared_layers...), Chain(policy_layers...), Chain(value_layers...))
end

function select_model_config(model_cfg, architecture::String)
    if haskey(model_cfg, "variants")
        variants = model_cfg["variants"]
        variant_cfg = get(variants, architecture, nothing)
        variant_cfg === nothing && throw(ArgumentError("Unsupported model architecture '$architecture'. Available variants: $(join(sort!(collect(keys(variants))), ", "))."))
        return variant_cfg
    end

    return model_cfg
end

function create_model(config_path::String=joinpath(@__DIR__, "config.toml"))
    config = TOML.parsefile(config_path)
    model_cfg = config["model"]
    architecture = model_architecture(model_cfg)
    selected_cfg = select_model_config(model_cfg, architecture)

    if architecture == DEFAULT_MODEL_ARCHITECTURE
        return build_mlp_model(selected_cfg)
    end

    throw(ArgumentError("Unsupported model architecture '$architecture'. Only 'mlp' is implemented in this checkpoint-safe slice."))
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

function atomic_write(write_fn::Function, path::AbstractString)
    parent = dirname(path)
    isdir(parent) || mkpath(parent)
    temp_path, io = mktemp(parent)
    success = false
    try
        write_fn(io)
        flush(io)
        close(io)
        mv(temp_path, path; force=true)
        success = true
    catch
        try
            isopen(io) && close(io)
        finally
            rethrow()
        end
    finally
        if !success && isfile(temp_path)
            rm(temp_path; force=true)
        end
    end
    return path
end

"""
    save_model(model, path)

Persist a checkpoint to `path` using Julia `Serialization`.

Checkpoint `.bin` files are a trusted-local artifact in this repo: they are expected to be
created by this project and loaded from the local workspace only.
"""
function save_model(model::AwaleModel, path::AbstractString)
    atomic_write(path) do io
        Serialization.serialize(io, model)
    end
    return path
end

"""
    load_model(path)

Load a checkpoint from `path` using Julia `Serialization`.

This is intentionally a trusted-local-only path for project-generated `.bin` files.
"""
function load_model(path::AbstractString)::AwaleModel
    open(path, "r") do io
        return Serialization.deserialize(io)
    end
end

end # module
