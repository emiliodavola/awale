# 04_neural_network: Network and encoding contract

This spec documents the **current** model/encoding contract implemented in `src/Awale/State.jl` and `src/Awale/Model.jl`.

## Quick path

1. Encode states with `encode_state(s)`.
2. Flatten the `4 x 12` tensor for the current MLP.
3. Interpret the 6 policy logits in local action order `1..6`.

## Input encoding

- `encode_state(s)::Matrix{Float32}` returns shape **`(4, 12)`**.
- Semantic planes:
  1. normalized board seeds
  2. side-to-move indicator plane
  3. normalized captured seeds for player 1
  4. normalized captured seeds for player 2
- Current MLP path flattens this encoding to **48 Float32 features**.

## Output contract

- Policy head returns `Vector{Float32}` of length **6**.
- Logits correspond to local actions **`1..6`**.
- Value head returns a scalar in `[-1, 1]`.

## Current implementation notes

| Area | Current contract |
|---|---|
| Architecture config | Loaded from local `src/Awale/config.toml` (template: `src/Awale/config.toml.example`) |
| Forward path | Shared trunk + policy/value heads in Flux |
| Policy activation | Final policy layer is unconstrained (`identity`) logits |
| Save/load | Uses Julia `Serialization.serialize` / `deserialize` |

## Determinism expectations

- `predict(model, s)` must be deterministic for fixed weights and state.
- `predict_batch` must preserve the same per-state semantics as `predict`.
- Serialization round-trips must preserve outputs for the same model binary.

## Explicit non-goals in the current repo state

- No convolutional architecture yet.
- No checkpoint metadata embedded with model binaries.
- No BSON/JLD2 save format at the moment.

## Testing checklist

- [ ] `encode_state` shape is `(4, 12)`
- [ ] policy head exposes raw logits
- [ ] identical canonical states produce identical outputs
- [ ] model save/load preserves predictions
