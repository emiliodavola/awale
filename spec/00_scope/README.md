00_scope: Goals and project scope

Goals
- Build a research-grade, reproducible, deterministic Awale RL system in Julia.
- Follow specification-driven development: specs → interfaces → property tests → implementation.
- Prioritize correctness, determinism, mathematical consistency, modularity, and testability.

Out-of-scope (initially)
- Large transformer models or GANs
- Complex neural architectures beyond small MLPs
- GPU-specific optimizations until correctness and tests exist

Deliverables
- Formal specs (this directory)
- Pure, deterministic environment API
- Property test suite proving invariants
- Modular components: rules, env, search, NN, training, eval

Decision log
- All design decisions must be recorded in /spec/11_research/decision_log.md