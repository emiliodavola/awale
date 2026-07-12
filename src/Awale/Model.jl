module Model

using TOML
using Flux
using Serialization
using ..State: GameState, canonicalize, encode_state
using ..Utils: fnv1a64

export create_model, predict, predict_batch, predict_raw, encode_state, save_model, load_model

const DEFAULT_MODEL_ARCHITECTURE = "mlp"
const CNN_SHARED_FEATURES = 128
const CNN_CONV1_CHANNELS = 8
const CNN_CONV2_CHANNELS = 16

mutable struct AwaleModel
    shared::Chain
    policy::Chain
    value::Chain
end

# Use @layer for Flux >= 0.15 to avoid deprecation warnings and ensure parameter tracking
Flux.@layer AwaleModel

struct ReshapeToCNN end
(layer::ReshapeToCNN)(x::AbstractMatrix{Float32}) = reshape(x, 4, 12, 1, size(x, 2))

struct FlattenBatch end
(layer::FlattenBatch)(x) = reshape(x, :, size(x, ndims(x)))

function model_architecture(model_cfg)::String
    architecture = get(model_cfg, "architecture", DEFAULT_MODEL_ARCHITECTURE)
    return lowercase(String(architecture))
end

function activation_map()
    return Dict(
        "relu" => relu,
        "tanh" => tanh,
        "identity" => identity,
    )
end

function build_dense_chain(layer_specs, act_map)
    return Chain([Dense(layer["in"] => layer["out"], act_map[layer["activation"]]) for layer in layer_specs]...)
end

function build_mlp_model(model_cfg)
    act_map = activation_map()

    haskey(model_cfg, "layers") || throw(ArgumentError("Model configuration is missing the 'layers' section."))
    layers = model_cfg["layers"]
    haskey(layers, "shared") || throw(ArgumentError("Model configuration is missing the 'shared' layer stack."))
    haskey(layers, "policy") || throw(ArgumentError("Model configuration is missing the 'policy' layer stack."))
    haskey(layers, "value") || throw(ArgumentError("Model configuration is missing the 'value' layer stack."))

    shared_layers = build_dense_chain(layers["shared"], act_map)
    policy_layers = build_dense_chain(layers["policy"], act_map)
    value_layers = build_dense_chain(layers["value"], act_map)

    return AwaleModel(shared_layers, policy_layers, value_layers)
end

function build_cnn_model(model_cfg)
    act_map = activation_map()

    haskey(model_cfg, "layers") || throw(ArgumentError("Model configuration is missing the 'layers' section."))
    layers = model_cfg["layers"]
    haskey(layers, "policy") || throw(ArgumentError("Model configuration is missing the 'policy' layer stack."))
    haskey(layers, "value") || throw(ArgumentError("Model configuration is missing the 'value' layer stack."))

    shared_layers = Chain(
        ReshapeToCNN(),
        Conv((3, 3), 1 => CNN_CONV1_CHANNELS, relu; pad=1),
        Conv((3, 3), CNN_CONV1_CHANNELS => CNN_CONV2_CHANNELS, relu; pad=1),
        FlattenBatch(),
        Dense(CNN_CONV2_CHANNELS * 4 * 12 => CNN_SHARED_FEATURES, relu),
    )
    policy_layers = build_dense_chain(layers["policy"], act_map)
    value_layers = build_dense_chain(layers["value"], act_map)

    return AwaleModel(shared_layers, policy_layers, value_layers)
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
    elseif architecture == "cnn"
        return build_cnn_model(selected_cfg)
    end

    throw(ArgumentError("Unsupported model architecture '$architecture'. Only 'mlp' and 'cnn' are implemented in this checkpoint-safe slice."))
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
