# Copilot instructions for this repository

Repository purpose
------------------
Research-grade Awale (Oware/Awari) reinforcement-learning system implemented in Julia. Specification-driven workflow: all design, invariants, and contracts live under spec/. Follow spec/ before editing core logic.

Spec locations (authoritative)
- spec/README.md
- spec/00_scope through spec/11_research (detailed specifications, invariants, and testing strategy)

1) Build, test, and lint commands (Julia)
----------------------------------------
Prerequisites: Julia (recommended stable release matching Project.toml). Use the project environment for deterministic dependency resolution.

- Install dependencies / instantiate environment:
  julia --project=. -e "using Pkg; Pkg.instantiate()"

- Run full test suite (standard):
  julia --project=. -e "using Pkg; Pkg.test()"

- Run a single test file (example):
  julia --project=. test/test_state_transitions.jl

- Run a single test by name from the REPL (example):
  julia --project=. -e "using Test; include(\"test/test_state_transitions.jl\"); # run specific @testset or @test" 

- Format code with JuliaFormatter:
  julia --project=. -e "using Pkg; Pkg.instantiate(); using JuliaFormatter; format(\"src\"; recursive=true)"

- Run benchmarks (if bench scripts provided):
  julia --project=. bench/run_benchmarks.jl

Notes:
- Prefer running tests under the project environment to ensure reproducible dependency versions.
- Single-test examples assume test files exist under test/*. Adjust paths to concrete test files.

2) High-level architecture (overview for Copilot)
------------------------------------------------
- Language: Julia (Flux.jl for networks; StaticArrays.jl for compact board representation)
- Top-level modules (recommended):
  - Awale.Rules  — rule definitions, variant flags, move semantics
  - Awale.State  — GameState type, canonicalize, serialize/deserialize, invariants
  - Awale.Env    — pure transition(s), legal_actions, terminal/reward API
  - Awale.Search — MCTS (PUCT), transposition table, node pool
  - Awale.Model  — Flux model (policy+value), batching helpers, model IO
  - Awale.Train  — self-play, replay buffer, training loop, checkpointing
  - Awale.Eval   — baselines, tournaments, ELO estimation
  - Awale.Utils  — RNG injection, hashing, deterministic helpers

Entry points:
- src/ (library code)
- test/ (property and unit tests)
- bench/ (benchmarks)

3) Key codebase conventions (important)
---------------------------------------
These are enforced by specs in spec/02_state_model and spec/01_game_rules.
- State representation: board :: SVector{12,UInt8} (StaticArrays), to_move ∈ {1,2}, captured :: NTuple{2,UInt8}
- Canonicalization: canonicalize(s) returns state from current-player perspective and must be idempotent
- Immutability: GameState is immutable; transition(s,a) returns new GameState and never mutates inputs
- Determinism: All transitions, serialization, hashing, and MCTS must be deterministic given same RNG and model outputs
- Invariants (always enforced):
  - Seed conservation: sum(board) + captured[1] + captured[2] == 48
  - Non-negativity: ∀i, board[i] ≥ 0
  - Serialization roundtrip: deserialize(serialize(s)) == s
- MCTS specifics:
  - Use AlphaZero-style PUCT: U = Q + c_puct * P * sqrt(N) / (1 + N_a)
  - Backup sign inversion: v_parent = -v_child (mandatory)
  - Priors from network are over local actions 0..5 (map to absolute indices via canonicalization)
  - Root-only Dirichlet noise with injected RNG for reproducibility
- Neural network:
  - Small MLP policy+value head (Flux.jl); input is canonicalized state encoding; outputs: policy logits for 6 local actions and scalar value in [-1,1]
- RNGs: inject RNGs into any non-deterministic routine (dirichlet noise, action sampling, shuffling). Do not use global RNG implicitly.

4) Testing and property-driven workflow
--------------------------------------
- The spec/09_testing directory contains the required property tests and fixtures. Implement properties before implementing the engine.
- Critical property tests to implement first:
  - seed conservation after any legal transition
  - transition purity (input state unchanged)
  - canonicalize idempotence
  - legal_actions respects variant flags (starvation, forced feeding)
  - serialize/deserialize roundtrip
- Determinism tests: fixed RNG + stubbed network outputs → identical MCTS distributions

5) Developer notes for Copilot sessions
--------------------------------------
- Always consult spec/ before proposing changes to core logic; specs are authoritative.
- When generating Julia code:
  - include full type annotations for public APIs
  - prefer StaticArrays and immutable structs for hot-path types
  - avoid hidden mutations and magic constants; surface variant flags via GameConfig
- Commit changes that touch specs alongside implementations updating them, if any behavior deviates from spec

Where to start
- Read spec/02_state_model/README.md and spec/01_game_rules/README.md to implement state and rules contracts
- Implement property tests in test/ per spec/09_testing before writing transition logic

Last-updated: 2026-05-27

Version control and branching (gitflow)
--------------------------------------
- Versioning model: gitflow.
- Permanent branches: `main` (stable releases) and `dev` (integration branch).
- Branch naming conventions:
  - feature branches: `feature/NAME`
  - release branches: `release/X.Y.Z`
  - hotfix branches: `hotfix/issue-description`
- Pull requests should be targeted at `dev`; merges to `main` should follow release testing and tagging.

MCP servers configuration (recommended)
--------------------------------------
- This project recommends configuring two MCP servers for benchmarks and GPU inference gateways. Example config is provided in `.mcp/servers.yml`.
- Servers to consider (examples):
  - `benchmark` — for running reproducible performance suites and microbenchmarks (CI/bench worker).
  - `gpu-inference` — for batched GPU inference when available (pinned CUDA driver and deterministic libs).

Repository initialization
-------------------------
- Repository is not yet initialized. The initial commit created by Copilot includes the spec/ directory and CI workflows.
- I will NOT set local git user.name or user.email automatically. Please set your preferred identity before I create commits, or grant explicit permission to set a provided name and email.
- If authorized, Copilot can create a private GitHub repository named `awale-rl` and push the initial commit. Remote name: `origin`.

If you want changes to tone, additional commands, or coverage for CI workflows, say so and I will update this file.