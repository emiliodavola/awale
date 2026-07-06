# Spec: Phase 1 - State Encoding

## Goal

Transition from a flat representation of `GameState` to a structured tensor of shape `(C, 12)` to enable convolutional processing in future architectures.

## Requirements

### 1. Tensor Shape

- Output format: `Matrix{Float32}(C, 12)`.
- Total Channels ($C$): 4.

### 2. Channel Definitions

- **Channel 1 (Seeds):** Normalized values of the 12 pits. `val[i] = board[i] / 48.0`.
- **Channel 2 (Active Player):** Indicator of the current player's position/turn. Suggested: A scalar signal where `1.0` represents Player 1 and `0.0` represents Player 2, or a one-hot representation.
- **Channel 3 (Captures A):** Total captures for Player 1 (as a single value broadcasted or as part of the state).
- **Channel 4 (Captures B):** Total captures for Player 2.

### 3. Constraints & Invariants

- **Normalization:** All values in the tensor must be in the range `[0.0, 1.0]`.
- **Data Integrity:** The sum of all values in `Channel 1` + `Channel 3` + `Channel 4` must always equal `1.0` (representing the total 48 seeds).
- **Idempotency:** `canonicalize(s)` must be applied before encoding to ensure the tensor representation is invariant to board rotation.

## Implementation Notes (for `src/Awale/Model.jl`)

- `encode_state(s::GameState)` must be updated to return a `Matrix{Float32}(4, 12)`.
- The `shared` layer in `AwaleModel` must be updated to accept this new input shape.
