"""
    MCTS

AlphaZero-style Monte Carlo Tree Search with PUCT selection,
Dirichlet root noise, and transposition-aware backup.
"""
module MCTS

using ..State: GameState, canonicalize, serialize_state
using ..Env: legal_actions, transition, is_terminal, reward
using ..Model: predict_inference
using ..Utils: fnv1a64
using Random
using Flux: softmax

export MCTSSearch, search, search_with_stats

"""
    MCTSNode

Single node in the MCTS tree.

# Fields
- state::GameState — the game position at this node
- prior::Float32 — prior probability from the policy network
- visits::Ref{Int64} — visit count (mutable via Ref for tree traversal)
- value_sum::Ref{Float32} — accumulated value (mutable via Ref)
- children::Dict{Int, MCTSNode} — child nodes keyed by action
"""
struct MCTSNode
    state::GameState
    prior::Float32
    visits::Ref{Int64}
    value_sum::Ref{Float32}
    children::Dict{Int, MCTSNode}
end

"""
    MCTSNode(s::GameState, prior::Float32=1.0f0) -> MCTSNode

Convenience constructor that initializes an unvisited MCTS node (zero visits, zero value sum)
with an empty children dictionary.
"""
function MCTSNode(s::GameState, prior::Float32=1.0f0)
    return MCTSNode(s, prior, Ref(0), Ref(0.0f0), Dict())
end

"""
    MCTSSearch

Search configuration holding the neural model and PUCT constant.

# Fields
- model — the neural network for policy/value predictions
- c_puct::Float32 — exploration constant in PUCT formula
- transposition_table::Dict — cache of (q_value, visits) by transposition key
"""
struct MCTSSearch
    model
    c_puct::Float32
    dirichlet_alpha::Float32
    dirichlet_epsilon::Float32
    transposition_table::Dict{UInt64, Tuple{Float64, Int64}}
end

"""
    transposition_key(state::GameState) -> UInt64

Compute a hash key that includes past history, enabling transposition-aware search.
"""
function transposition_key(state::GameState)::UInt64
    bytes = serialize_state(state)
    for past_hash in sort!(collect(state.history_hashes))
        for shift in 0:8:56
            push!(bytes, UInt8((past_hash >> shift) & 0xff))
        end
    end
    return fnv1a64(bytes)
end

"""
    search(mcts::MCTSSearch, root_state, num_sims, rng; add_root_noise) -> Int

Run MCTS and return the best action. Convenience wrapper around `search_with_stats`.
"""
function search(mcts::MCTSSearch, root_state::GameState, num_sims::Int, rng=Random.default_rng(); add_root_noise::Bool=false)
    action, _ = search_with_stats(mcts, root_state, num_sims, rng; add_root_noise=add_root_noise)
    return action
end

"""
    legal_action_priors(logits, actions) -> Vector{Float32}

Mask policy logits to legal actions and softmax-normalize them into priors.
"""
function legal_action_priors(logits::AbstractVector{<:Real}, actions::Vector{Int})
    mask = fill(-Inf32, length(logits))
    for action in actions
        mask[action] = Float32(logits[action])
    end

    probs = softmax(mask)
    priors = Float32[probs[action] for action in actions]
    total = sum(priors)

    if total <= 1.0f-8
        return fill(1.0f0 / length(actions), length(actions))
    end

    return priors ./ total
end

"""
    search_with_stats(mcts, root_state, num_sims, rng; add_root_noise) -> (best_action, policy)

Run MCTS for `num_sims` iterations, returning the best action and visit-count policy.
When `add_root_noise` is true, Dirichlet noise is added to root priors for exploration.
"""
function search_with_stats(
    mcts::MCTSSearch,
    root_state::GameState,
    num_sims::Int,
    rng=Random.default_rng();
    add_root_noise::Bool=false,
)
    empty!(mcts.transposition_table)

    root = MCTSNode(canonicalize(root_state))
    actions = legal_actions(root.state)
    if isempty(actions)
        return 0, zeros(Float32, 6)
    end

    logits, _ = predict_inference(mcts.model, root.state)
    root_priors = legal_action_priors(vec(logits), actions)

    if add_root_noise
        dir_noise = generate_dirichlet(rng, length(actions), mcts.dirichlet_alpha)
        root_priors = ((1.0f0 - mcts.dirichlet_epsilon) .* root_priors) .+ (mcts.dirichlet_epsilon .* dir_noise)
        root_priors ./= sum(root_priors)
    end

    for (idx, action) in enumerate(actions)
        next_state = transition(root.state, action)
        root.children[action] = MCTSNode(canonicalize(next_state), root_priors[idx])
    end

    if num_sims <= 0
        policy = zeros(Float32, 6)
        for (idx, action) in enumerate(actions)
            policy[action] = root_priors[idx]
        end
        best_action = actions[argmax(root_priors)]
        return best_action, policy
    end

    for _ in 1:num_sims
        leaf, path = select_and_expand(mcts, root)
        leaf_value = if is_terminal(leaf.state)
            reward(leaf.state)
        else
            _, value = predict_inference(mcts.model, leaf.state)
            value
        end
        backup(path, leaf_value, mcts)
    end

    counts = zeros(Float32, 6)
    total_visits = sum(node.visits[] for node in values(root.children))
    for (action, node) in root.children
        counts[action] = node.visits[] / max(1, total_visits)
    end

    best_action = argmax(counts)
    return best_action, counts
end

"""
    generate_dirichlet(rng, n, alpha) -> Vector{Float32}

Sample an n-dimensional Dirichlet distribution with concentration parameter `alpha`.
"""
function generate_dirichlet(rng, n, alpha)
    samples = Float32[]
    for _ in 1:n
        push!(samples, sample_gamma(rng, alpha))
    end
    total = sum(samples)
    return samples ./ total
end

"""
    sample_gamma(rng, alpha) -> Float32

Sample from a Gamma(alpha, 1) distribution using the Marsaglia–Tsang algorithm.
Used by `generate_dirichlet` to produce Dirichlet noise for MCTS root exploration.
"""
function sample_gamma(rng, alpha)
    if alpha < 1.0f0
        return sample_gamma(rng, alpha + 1.0f0) * (Float32(rand(rng))^(1.0f0 / alpha))
    end
    d = alpha - 1.0f0 / 3.0f0
    c = 1.0f0 / sqrt(9.0f0 * d)
    while true
        x = Float32(randn(rng))
        v = (1.0f0 + c * x)^3
        if v <= 0
            return sample_gamma(rng, alpha)
        end
        u = Float32(rand(rng))
        if u < 1.0f0 - 0.0331f0 * x^4
            return d * v
        end
        if log(u) < 0.5f0 * x^2 + d * (1.0f0 - v + log(v))
            return d * v
        end
    end
end

"""
    select_and_expand(mcts, root) -> (leaf, path)

Traverse the tree using PUCT, expanding leaf nodes when encountering
unexpanded or terminal positions.
"""
function select_and_expand(mcts::MCTSSearch, root::MCTSNode)
    path = MCTSNode[]
    current = root
    while true
        push!(path, current)
        if is_terminal(current.state)
            return current, path
        end
        if isempty(current.children)
            expand(mcts, current)
            return current, path
        end
        action = select_puct(mcts, current)
        current = current.children[action]
    end
end

"""
    expand(mcts, node)

Expand a leaf node by evaluating it with the neural network and creating
child nodes for all legal actions.
"""
function expand(mcts::MCTSSearch, node::MCTSNode)
    actions = legal_actions(node.state)
    isempty(actions) && return

    logits, _ = predict_inference(mcts.model, node.state)
    priors = legal_action_priors(vec(logits), actions)

    for (idx, action) in enumerate(actions)
        next_state = transition(node.state, action)
        child_state = canonicalize(next_state)
        node.children[action] = MCTSNode(child_state, priors[idx])
    end
end

"""
    select_puct(mcts, node) -> Int

Select the child action with the highest PUCT score:
    U = Q + c_puct * P * sqrt(N_parent) / (1 + N_child)

Transposition table hits use cached Q/visits instead of the raw child values.
"""
function select_puct(mcts::MCTSSearch, node::MCTSNode)
    best_score = -Float32(Inf)
    best_action = 0
    parent_visits = max(1, node.visits[])

    for (action, child) in node.children
        q_value = child.value_sum[] / max(1, child.visits[])
        child_visits = child.visits[]
        child_key = transposition_key(child.state)

        if haskey(mcts.transposition_table, child_key)
            cached_q, cached_visits = mcts.transposition_table[child_key]
            score = -cached_q + mcts.c_puct * child.prior * sqrt(parent_visits) / (1 + cached_visits)
        else
            score = -q_value + mcts.c_puct * child.prior * sqrt(parent_visits) / (1 + child_visits)
        end

        if score > best_score
            best_score = score
            best_action = action
        end
    end

    return best_action
end

"""
    backup(path, leaf_value, mcts)

Backpropagate the leaf evaluation through the search path with sign inversion
at each level (standard AlphaZero backup).
Updates the transposition table with averaged Q and visit counts.
"""
function backup(path::Vector{MCTSNode}, leaf_value::Float32, mcts::MCTSSearch)
    value = leaf_value
    for idx in length(path):-1:1
        node = path[idx]
        node.visits[] += 1
        node.value_sum[] += value

        average_q = node.value_sum[] / node.visits[]
        mcts.transposition_table[transposition_key(node.state)] = (average_q, node.visits[])
        value = -value
    end
end

end # module