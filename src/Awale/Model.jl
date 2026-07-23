"""
    Model

Neural network model definition, inference, checkpoint persistence, and
public model export for Hugging Face releases.
"""
module Model

using TOML
using Flux
using Serialization
using ..State: GameState, canonicalize, encode_state

export create_model, predict, predict_batch, predict_inference, predict_batch_inference, predict_raw, encode_state, save_model, load_model, save_public_model, load_public_model

"""
    AwaleModel

Dual-head neural network with shared trunk, policy head (logits), and value head (scalar).
Uses `Flux.@layer` for automatic parameter tracking.
"""
mutable struct AwaleModel
    shared::Chain
    policy::Chain
    value::Chain
end

# Use @layer for Flux >= 0.15 to avoid deprecation warnings and ensure parameter tracking
Flux.@layer AwaleModel

"""
    ReshapeLayer

Flux-compatible layer that reshapes input to a fixed target shape.
"""
struct ReshapeLayer
    shape::Vector{Int}
end

(layer::ReshapeLayer)(x) = reshape(x, Tuple(layer.shape)..., size(x, ndims(x)))

struct FlattenLayer end
(layer::FlattenLayer)(x) = reshape(x, :, size(x, ndims(x)))

struct GlobalAveragePoolLayer end

function global_average_pool_dims(x)
    ndims(x) <= 2 && return (1,)
    return ntuple(identity, ndims(x) - 2)
end

function global_average_pool(x)
    dims = global_average_pool_dims(x)
    denom = one(eltype(x))
    for d in dims
        denom *= size(x, d)
    end
    return sum(x; dims=dims) ./ denom
end

(layer::GlobalAveragePoolLayer)(x) = global_average_pool(x)

"""
    model_architecture(model_cfg) -> String

Extract the architecture name from a parsed TOML model configuration.
"""
function model_architecture(model_cfg)::String
    architecture = get(model_cfg, "architecture", "mlp")
    return lowercase(String(architecture))
end

function activation_map()
    return Dict(
        "identity" => identity,
        "relu" => relu,
        "tanh" => tanh,
        "sigmoid" => sigmoid,
    )
end

function parse_activation(spec, act_map)
    activation_name = lowercase(String(get(spec, "activation", "identity")))
    activation = get(act_map, activation_name, nothing)
    activation === nothing && throw(ArgumentError("Unsupported activation '$activation_name'. Available activations: $(join(sort!(collect(keys(act_map))), ", "))."))
    return activation
end

function normalize_type_name(type_name)
    return replace(lowercase(String(type_name)), "_" => "", "-" => "")
end

function as_int_tuple(value, field_name::AbstractString)
    if value isa Integer
        return (Int(value),)
    end

    value isa AbstractVector || value isa Tuple || throw(ArgumentError("Layer configuration field '$field_name' must be an integer or an integer tuple/array."))
    values = Tuple(Int.(collect(value)))
    isempty(values) && throw(ArgumentError("Layer configuration field '$field_name' must not be empty."))
    return values
end

function as_repeated_int_tuple(value, dims::Int, field_name::AbstractString)
    if value isa Integer
        return ntuple(_ -> Int(value), dims)
    end

    values = as_int_tuple(value, field_name)
    length(values) == dims || throw(ArgumentError("Layer configuration field '$field_name' must have exactly $dims entries."))
    return values
end

function dense_layer(spec, act_map)
    haskey(spec, "in") || throw(ArgumentError("Dense layer is missing the 'in' field."))
    haskey(spec, "out") || throw(ArgumentError("Dense layer is missing the 'out' field."))
    return Dense(Int(spec["in"]) => Int(spec["out"]), parse_activation(spec, act_map))
end

function conv_layer(spec::AbstractDict, act_map)
    kernel = get(spec, "kernel", get(spec, "size", nothing))
    kernel === nothing && throw(ArgumentError("Conv layer is missing the 'kernel' field."))
    kernel_tuple = as_int_tuple(kernel, "kernel")
    haskey(spec, "in") || throw(ArgumentError("Conv layer is missing the 'in' field."))
    haskey(spec, "out") || throw(ArgumentError("Conv layer is missing the 'out' field."))
    activation = parse_activation(spec, act_map)
    stride = as_repeated_int_tuple(get(spec, "stride", 1), length(kernel_tuple), "stride")
    pad = as_repeated_int_tuple(get(spec, "pad", 0), length(kernel_tuple), "pad")
    dilation = as_repeated_int_tuple(get(spec, "dilation", 1), length(kernel_tuple), "dilation")
    groups = Int(get(spec, "groups", 1))
    return Conv(kernel_tuple, Int(spec["in"]) => Int(spec["out"]), activation; stride=stride, pad=pad, dilation=dilation, groups=groups)
end

function reshape_layer(spec::AbstractDict)
    shape = get(spec, "shape", get(spec, "dims", nothing))
    shape === nothing && throw(ArgumentError("Reshape layer is missing the 'shape' field."))
    return ReshapeLayer(collect(as_int_tuple(shape, "shape")))
end

flatten_layer(::AbstractDict) = FlattenLayer()

function maxpool_layer(spec::AbstractDict)
    size = get(spec, "size", get(spec, "kernel", nothing))
    size === nothing && throw(ArgumentError("MaxPool layer is missing the 'size' field."))
    size_tuple = as_int_tuple(size, "size")
    stride = as_repeated_int_tuple(get(spec, "stride", size_tuple), length(size_tuple), "stride")
    pad = as_repeated_int_tuple(get(spec, "pad", 0), length(size_tuple), "pad")
    return MaxPool(size_tuple; stride=stride, pad=pad)
end

function meanpool_layer(spec::AbstractDict)
    size = get(spec, "size", get(spec, "kernel", nothing))
    size === nothing && throw(ArgumentError("MeanPool layer is missing the 'size' field."))
    size_tuple = as_int_tuple(size, "size")
    stride = as_repeated_int_tuple(get(spec, "stride", size_tuple), length(size_tuple), "stride")
    pad = as_repeated_int_tuple(get(spec, "pad", 0), length(size_tuple), "pad")
    return MeanPool(size_tuple; stride=stride, pad=pad)
end

function global_average_pool_layer(::AbstractDict)
    return GlobalAveragePoolLayer()
end

function batchnorm_layer(spec::AbstractDict, act_map)
    size = get(spec, "size", get(spec, "channels", nothing))
    size === nothing && throw(ArgumentError("BatchNorm layer is missing the 'size' field."))
    activation = parse_activation(spec, act_map)
    affine = get(spec, "affine", true)
    track_stats = get(spec, "track_stats", true)
    return BatchNorm(Int(size), activation; affine=affine, track_stats=track_stats)
end

function dropout_layer(spec::AbstractDict)
    rate = get(spec, "rate", get(spec, "p", nothing))
    rate === nothing && throw(ArgumentError("Dropout layer is missing the 'rate' field."))
    return Dropout(Float64(rate))
end

"""
    build_layer(spec, act_map)

Build a single Flux layer from a TOML specification dictionary.
Supports: Dense, Conv, Reshape, Flatten, MaxPool, MeanPool, GlobalAveragePool, BatchNorm, Dropout.
"""
function build_layer(spec::AbstractDict, act_map)
    haskey(spec, "type") || throw(ArgumentError("Layer specification is missing the 'type' field."))
    layer_type = normalize_type_name(spec["type"])

    if layer_type == "dense"
        return dense_layer(spec, act_map)
    elseif layer_type == "conv"
        return conv_layer(spec, act_map)
    elseif layer_type == "reshape"
        return reshape_layer(spec)
    elseif layer_type == "flatten"
        return flatten_layer(spec)
    elseif layer_type == "maxpool"
        return maxpool_layer(spec)
    elseif layer_type == "meanpool"
        return meanpool_layer(spec)
    elseif layer_type == "globalaveragepool" || layer_type == "globalmeanpool"
        return global_average_pool_layer(spec)
    elseif layer_type == "batchnorm"
        return batchnorm_layer(spec, act_map)
    elseif layer_type == "dropout"
        return dropout_layer(spec)
    end

    throw(ArgumentError("Unsupported layer type '$layer_type'. Supported types: Dense, Conv, Reshape, Flatten, MaxPool, MeanPool, GlobalAveragePool, BatchNorm, Dropout."))
end

function build_layer_stack(layer_specs, act_map, stack_name::AbstractString)
    layer_specs isa AbstractVector || throw(ArgumentError("Model configuration layer stack '$stack_name' must be an array of layer specifications."))
    isempty(layer_specs) && throw(ArgumentError("Model configuration layer stack '$stack_name' must not be empty."))
    return Chain([build_layer(layer, act_map) for layer in layer_specs]...)
end

"""
    build_model(model_cfg) -> AwaleModel

Construct an `AwaleModel` from a parsed TOML configuration section.
Expects `shared`, `policy`, and `value` layer stacks.
"""
function build_model(model_cfg)
    haskey(model_cfg, "layers") || throw(ArgumentError("Model configuration is missing the 'layers' section."))
    layers = model_cfg["layers"]
    haskey(layers, "shared") || throw(ArgumentError("Model configuration is missing the 'shared' layer stack."))
    haskey(layers, "policy") || throw(ArgumentError("Model configuration is missing the 'policy' layer stack."))
    haskey(layers, "value") || throw(ArgumentError("Model configuration is missing the 'value' layer stack."))

    act_map = activation_map()
    shared_layers = build_layer_stack(layers["shared"], act_map, "shared")
    policy_layers = build_layer_stack(layers["policy"], act_map, "policy")
    value_layers = build_layer_stack(layers["value"], act_map, "value")

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

"""
    create_model(config_path::String=...) -> AwaleModel

Load model configuration from TOML and build the neural network.
Reads architecture selection and variant config from the file.
"""
function create_model(config_path::String=joinpath(@__DIR__, "config.toml"))
    config = TOML.parsefile(config_path)
    model_cfg = config["model"]
    architecture = model_architecture(model_cfg)
    selected_cfg = select_model_config(model_cfg, architecture)
    return build_model(selected_cfg)
end

"""
    predict_raw(model::AwaleModel, X::AbstractMatrix{Float32}) -> (logits, value)

Run forward pass through shared trunk, policy head, and value head.
Returns policy logits and a scalar value estimate.
"""
function predict_raw(model::AwaleModel, X::AbstractMatrix{Float32})
    shared_out = model.shared(X)
    logits = model.policy(shared_out)
    value = model.value(shared_out)
    return logits, value
end

function with_inference_mode(f::Function, model::AwaleModel)
    Flux.testmode!(model)
    try
        return f()
    finally
        Flux.trainmode!(model)
    end
end

"""
    predict_inference(model::AwaleModel, s::GameState) -> (logits, value)

Run inference on a single state, switching the model to test mode temporarily.
"""
function predict_inference(model::AwaleModel, s::GameState)
    s_can = canonicalize(s)
    x = reshape(vec(encode_state(s_can)), :, 1)
    return with_inference_mode(() -> begin
        logits, value = predict_raw(model, x)
        return vec(logits), value[1]
    end, model)
end

"""
    predict_batch_inference(model::AwaleModel, states::Vector{GameState}) -> (logits, values)

Run batched inference with test mode, returning raw logits and values.
"""
function predict_batch_inference(model::AwaleModel, states::Vector{GameState})
    X = hcat([vec(encode_state(canonicalize(s))) for s in states]...)
    return with_inference_mode(() -> predict_raw(model, X), model)
end

"""
    predict(model::AwaleModel, s::GameState) -> (logits, value)

Forward pass on a single state in train mode. Prefer `predict_inference` for inference.
"""
function predict(model::AwaleModel, s::GameState)
    s_can = canonicalize(s)
    x = reshape(vec(encode_state(s_can)), :, 1)
    logits, value = predict_raw(model, x)
    return vec(logits), value[1]
end

"""
    predict_batch(model::AwaleModel, states::Vector{GameState}) -> (logits, values)

Batched forward pass in train mode.
"""
function predict_batch(model::AwaleModel, states::Vector{GameState})
    X = hcat([vec(encode_state(canonicalize(s))) for s in states]...)
    return predict_raw(model, X)
end

function predict_inference(model, s::GameState)
    return predict(model, s)
end

function predict_batch_inference(model, states::Vector{GameState})
    return predict_batch(model, states)
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

function public_model_weights(model::AwaleModel)::Vector{Float32}
    weights, _ = Flux.destructure(model)
    return Float32.(weights)
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
    save_public_model(model, path)

Persist a release-safe checkpoint payload containing only raw `Float32` weights.

This is a public-export format for Hugging Face releases: it is data-only, not a Julia
serialized object, and must be reconstructed with `load_public_model`.
"""
function save_public_model(model::AwaleModel, path::AbstractString)
    weights = public_model_weights(model)
    atomic_write(path) do io
        write(io, weights)
    end
    return path
end

function public_model_config_path(path::AbstractString)::String
    return joinpath(dirname(String(path)), "model_config.toml")
end

"""
    load_public_model(path[, config_path])

Load a public release checkpoint saved by `save_public_model`.

By default, the model structure is recreated from a sibling `model_config.toml`
next to the public checkpoint. An explicit `config_path` still overrides that
inferred location.
"""
function load_public_model(path::AbstractString, config_path::AbstractString=public_model_config_path(path))::AwaleModel
    filesize(path) % sizeof(Float32) == 0 || throw(ArgumentError("Invalid public model payload size: $path"))
    weights = Vector{Float32}(undef, filesize(path) ÷ sizeof(Float32))
    open(path, "r") do io
        read!(io, weights)
    end

    prototype = create_model(String(config_path))
    _, reconstruct = Flux.destructure(prototype)
    return reconstruct(weights)
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