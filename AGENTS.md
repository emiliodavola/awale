# Repository Guidelines

This document serves as the authoritative guide for contributors to ensure consistency, quality, and strict determinism in the Awale RL system.

## Project Structure & Architecture

The project follows a specification-driven workflow. **Always consult the `spec/` directory (from `00_scope` to `11_research`) before modifying core logic.**

### Module Organization (`src/`)

- `State.jl`: Immutable state, canonicalization, and serialization.
- `Env.jl`: Game rules, transitions, and reward API.
- `Model.jl`: Flux.jl neural network (policy+value heads).
- `MCTS.jl`: AlphaZero-style search (PUCT) and node management.
- `Training.jl`: Self-play loops and model optimization.
- `Utils.jl`: Deterministic helpers, hashing, and RNG injection.

## Build, Test, and Development Commands

Use the project environment (`--project=.`) for all operations to ensure reproducible dependencies.

**Important Shell Quoting**: When running Julia code via `-e`, use **single quotes** (`'`) for the shell wrapper and **double quotes** (`"`) for internal Julia strings to avoid escaping issues: `julia --project=. 'using Pkg; Pkg.add("PackageName")'`.

- **Environment Setup**: `julia --project=. -e 'using Pkg; Pkg.instantiate()'`
- **Run All Tests**: `julia --project=. -e 'using Pkg; Pkg.test()'` or `julia test/runtests.jl`
- **Start Training**: `julia --project=. train.jl`
- **Format Code**: `julia --project=. -e 'using JuliaFormatter; format("src"; recursive=true)'`

## Core Technical Constraints & Invariants

To maintain research-grade reliability, the following must be strictly enforced:

- **Game Invariants**:
  - Seed conservation: $\sum (\text{board}) + \text{captured}_1 + \text{captured}_2 = 48$.
  - Non-negativity: All pits $\ge 0$.
  - Serialization: `deserialize(serialize(s)) == s`.
- **MCTS Search**:
  - Must use PUCT: $U = Q + c_{puct} \cdot P \cdot \frac{\sqrt{N}}{1 + N_a}$.
  - Mandatory backup sign inversion: $v_{\text{parent}} = -v_{\text{child}}$.
  - Root exploration must use Dirichlet noise with injected RNG.
- **Neural Network**: Input is a canonicalized state encoding; outputs are policy logits for 6 local actions and a scalar value in $[-1, 1]$.

## GameState API Contract (CRITICAL)

The implementation of `GameState` must adhere to strict invariants to ensure test stability and determinism across module boundaries (`src/` vs `test/`). These rules govern state construction and consumption.

- **Canonical Definition:** The authoritative structure is defined by the public fields:

```julia
struct GameState{
    board::SVector{12, UInt8}
    to_move::Int8
    captured::NTuple{2, UInt8}
    history_hash::UInt64
    config::GameConfig
    history_hashes::Set{UInt64}
end
```

- **Constructor Policy:** The system MUST rely on a canonical full constructor and safe, outer convenience constructors for flexibility (e.g., `Vector` $\to$ `SVector`). Breaking existing constructor calls or removing old ones without migration is forbidden.
- **Single Source of Truth:** Any change to the underlying type structure (e.g., field addition/removal in `GameState`) requires a multi-step update:
  1) Update all constructors,
  2) Add compatibility layers if necessary, and
  3) Update ALL dependent tests. Never leave partial mismatches between source code and test files.
- **Testing Principle:** Tests must validate *behavior*, not construction mechanics. Factory functions (like `Awale.initial_state()`) should always be preferred over direct manual raw field initialization in tests.

- **Determinism**: All transitions, hashing, and search logic must be deterministic given the same RNG seed and model weights.
- **RNG Management**: Inject RNG objects into non-deterministic routines; **do not use the global RNG**.
- **Immutability**: `GameState` is immutable. Use functional updates (return new states).
- **Types**: Prefer `StaticArrays` for hot-path board representations and include full type annotations for public APIs.

## Julia & Flux Implementation Details

**Crucial Invariants for Development:**

- **No Method Overwriting**: Never define the same function signature twice within a module (e.g., two `predict(...)` functions). This will cause critical failures during Julia's module precompilation.
- **Encoding Consistency**: Always use the canonical `encode_state()` function to prepare inputs for the neural network. Manual normalization or bypassing this function leads to `DimensionMismatch` errors because the model architecture expects a fixed input size (14 features: 12 pits + 2 captured).
- Syntax Clarity: Do not apply Python syntax patterns to Julia. For example, avoid using the `*` operator for string repetition (use repeat() instead).

To ensure seamless integration with Zygote and Optimisers, follow these rules:

- **Model Definition**: Any custom model structure must use the `Flux.@functor` macro to enable parameter tracking for automatic differentiation.
- **Parameter Updates**:
  - If using a mutable model object, `Flux.update!(opt, model, grads)` updates in-place.
  - If using immutable structures (e.g., `NamedTuple`), the result of Flux.update! must be re-assigned: model = `Flux.update!(opt, model, grads)`.
- **Gradient Handling**: `Flux.gradient` returns a tuple containing gradients for each argument passed to the loss function. When training a single model, typically use the first element: `grads = Flux.gradient(loss_fn, model)[1]`.
- **Numeric Precision**: Use explicit Float32 for all neural network weights and inputs to avoid type instability and performance penalties associated with Julia's default Float64.

## Testing & Property Verification

Implement property tests in `test/` before implementing logic:

1. Seed conservation after any legal transition.
2. Transition purity (input state remains unchanged).
3. Canonicalization idempotence.
4. Serialization roundtrip.

## Gitflow & Contribution Guidelines

- **Branching Strategy**:
  - `main`: Stable releases.
  - `dev`: Integration branch for features.
  - `feature/NAME`, `hotfix/DESCRIPTION`, `release/X.Y.Z`.
- **Commit Messages**: Use Conventional Commits (`feat:`, `fix:`, `refactor:`, etc.).
- **PRs**: Target the `dev` branch. Include references to the modified `spec/` file and confirm that all property tests pass.
