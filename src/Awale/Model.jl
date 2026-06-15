module Model

using Flux
using ..State: GameState, canonicalize

export AwaleNet, create_model, predict, predict_batch, encode_state

"""
AwaleNet architecture as per spec/04_neural_network:
Input (14) -> Dense(128, relu) -> Dense(128, relu) -> 
    Policy Head: Dense(64, relu) -> Dense(6)
    Value Head: Dense(64, relu) -> Dense(1, tanh)
"""
struct AwaleNet
    shared_layers::Chain
    policy_head::Chain
    value_head::Chain
end

function create_model()
    shared = Chain(
        Dense(14 => 128, relu),
        Dense(128 => 128, relu)
    )
    
    policy = Chain(
        Dense(128 => 64, relu),
        Dense(64 => 6)
    )
    
    value = Chain(
        Dense(128 => 64, relu),
        Dense(64 => 1, tanh)
    )
    
    return AwaleNet(shared, policy, value)
end

# encode_state: converts GameState to a normalized Float32 vector (14 features)
function encode_state(s::GameState)::Vector{Float32}
    # Board normalization (/ 48)
    board_feats = [Float32(x) / 48.0f0 for x in s.board]
    # Captured scores normalization (/ 48)
    captured_feats = [Float32(s.captured[1]) / 48.0f0, Float32(s.captured[2]) / 48.0f0]
    
    return vcat(board_feats, captured_feats)
end

# predict: returns (policy_logits, value) for a single state
function predict(model::AwaleNet, s::GameState)
    s_can = canonicalize(s)
    x = encode_state(s_can)
    
    # Forward pass through shared layers
    shared_out = model.shared_layers(x)
    
    # Policy and Value heads
    logits = model.policy_head(shared_out)
    val = model.value_head(shared_out)[1] # scalar
    
    return (logits, val)
end

# predict_batch: handles a batch of states for efficiency
function predict_batch(model::AwaleNet, states::Vector{GameState})
    # Encode all states into a matrix [14 x BatchSize]
    X = hcat([encode_state(canonicalize(s)) for s in states]...)
    
    shared_out = model.shared_layers(X)
    logits_batch = model.policy_head(shared_out) # [6 x BatchSize]
    values_batch = model.value_head(shared_out)  # [1 x BatchSize]
    
    return (logits_batch, values_batch)
end

end # module
