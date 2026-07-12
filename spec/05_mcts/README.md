05_mcts: Search semantics, node structure, PUCT, and backup rules

Goals
- Implement AlphaZero-style MCTS with PUCT.
- Ensure backup inverts perspective (v_parent = -v_child).
- Maintain determinism given fixed RNG and deterministic network outputs.

Node definition (immutable view)
- struct MCTSNode
    state::GameState  # canonicalized view for node
    prior::Float32    # prior probability P(s,a)
    visits::Int64
    value_sum::Float32
    children::Dict{Int,MCTSNode}  # action->child
  end

Important derived quantities
- Q(s,a) = value_sum(s,a) / max(1, visits(s,a))
- N(s) = sum_a visits(s,a)

PUCT selection
- U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
- Selection chooses argmax_a U(s,a)

Algorithm outline
1. For i in 1..num_simulations:
   - selection: descend from root via argmax_a U(s,a) until leaf
   - expansion: if leaf is non-terminal and not previously expanded, call network to get prior and value, then create child nodes with priors
   - backup: v = value_from_network (from perspective of node.to_move). On backing up from child to parent, invert sign: v_parent_contribution = -v_child
   - increment visit counts and update value_sums accordingly

Contracts and determinism
- Given same root state, model outputs, c_puct, and RNG (for tie-breaking and dirichlet noise), MCTS must produce identical visit distributions.
- Network may be asked for predictions only at expansion time; cache predictions for identical canonicalized states.

Edge cases
- Terminal nodes: when is_terminal(s) true, expansion returns no priors; use reward(s) as leaf value
- Illegal moves must never appear in children; when network outputs priors for illegal actions, renormalize over legal actions before use

Exploration noise
- At root only, apply Dirichlet(alpha) and mixing weight epsilon to priors to promote exploration. Noise generation must use injected RNG for reproducibility.

Transposition handling
- Use canonicalized state hash as transposition table key. When encountering existing state in table, merge statistics or reuse node.
- Define explicit policy for merging duplicate nodes (e.g., sum visits, weighted priors).

Testable properties
- Backup inversion: after single backup, parent.value_sum == -child.value_sum contribution
- Visit distribution conservation: total visits at root equals number of simulations
- Determinism with fixed RNG and deterministic model outputs