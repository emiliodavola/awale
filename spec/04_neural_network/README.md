04_neural_network: Network design, inputs, outputs, and training contract

Design goals
- Small deterministic MLP-style policy+value network using Flux.jl
- Deterministic forward pass for a given input and RNG state (for dropout: avoid or disable in training/eval separation)
- Shared network for both players; inputs are canonicalized current-player views

Input encoding
- Input vector x ∈ R^{N} where N = 12 (board) + 2 (captured normalized) + optional feature flags
- Normalize board: divide by 48 or use fixed scaling to keep inputs in a numerically stable range
- Use Float32 tensors. For batching, use Matrix{Float32} with features on columns.

Output
- Policy logits y_policy ∈ R^{6} corresponding to local-side actions 0..5 (unused for illegal actions but network returns logits for all)
- Value scalar y_value ∈ [-1,1]

Suggested architecture (small)
- Input layer → Dense(128, relu) → Dense(128, relu) → (two heads)
  - policy_head: Dense(64, relu) → Dense(6)  # logits
  - value_head: Dense(64, relu) → Dense(1, tanh)
- Use small weight decay L2 regularization

Contracts
- predict(model, s_canonical)::(logits::Vector{Float32}, value::Float32)
  - Always interpret logits as corresponding to local indices 0..5
- model parameters saved via BSON.jl or JLD2 with deterministic metadata (seed, Flux/Julia versions)

Training-time determinism
- Control random seeds for weight init, data shuffling, and augmentation
- Avoid non-deterministic GPU ops until later phases; document any ops that may vary between runs

Performance
- Support batched inference: predict_batch(model, batch_states)
- Use StaticArrays for single-state encoding where beneficial, but convert to dense batches for Flux

Testable properties
- Predict on canonicalized identical states returns identical outputs
- Serialization/deserialization of model preserves outputs on sample inputs (within floating precision)