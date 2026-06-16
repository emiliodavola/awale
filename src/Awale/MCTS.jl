module MCTS

using ..State: GameState, canonicalize
using ..Env: legal_actions, transition, is_terminal, reward
using ..Model: predict
using Random
using Flux: softmax

export MCTSSearch, search, search_with_stats

struct MCTSNode
    state::GameState
    prior::Float32 
    visits::Ref{Int64}
    value_sum::Ref{Float32}
    children::Dict{Int, MCTSNode}
end

function MCTSNode(s::GameState, prior::Float32=1.0f0)
    return MCTSNode(s, prior, Ref(0), Ref(0.0f0), Dict())
end

struct MCTSSearch
    model
    c_puct::Float32
    # Transposition Table: Maps canonical state hash to the best known value (Q) and visit count (N) for optimization lookup
    transposition_table::Dict{UInt64, Tuple{Float32, Int64}} 
end

function search(mcts::MCTSSearch, root_state::GameState, num_sims::Int, rng=Random.default_rng())
    action, _ = search_with_stats(mcts, root_state, num_sims, rng)
    return action
end

function search_with_stats(mcts::MCTSSearch, root_state::GameState, num_sims::Int, rng=Random.default_rng())

    root = MCTSNode(canonicalize(root_state))

    actions = legal_actions(root.state)
    if isempty(actions)
        return 0, Float32[]
    end

    logits, _ = predict(mcts.model, root.state)
    logits = vec(logits)  # 🔥 obligatorio

    probs = softmax(logits)  # 🔥 SOLO ESTO

    dir_noise = generate_dirichlet(rng, length(probs), 0.3f0)
    epsilon = 0.25f0

    root_priors = (1f0 - epsilon) .* probs .+ epsilon .* dir_noise

    for action in actions
        s_next = transition(root.state, action)
        root.children[action] = MCTSNode(canonicalize(s_next), root_priors[action])
    end

    for _ in 1:num_sims
        leaf, path = select_and_expand(mcts, root)

        val = if is_terminal(leaf.state)
            reward(leaf.state)
        else
            _, v = predict(mcts.model, leaf.state)
            v
        end

        backup(path, val, mcts)
    end

    counts = zeros(Float32, 6)
    total_visits = sum(node.visits[] for node in values(root.children))

    for (action, node) in root.children
        counts[action] = node.visits[] / max(1, total_visits)
    end

    best_action = 0
    max_visits = -1

    for (action, node) in root.children
        if node.visits[] > max_visits
            max_visits = node.visits[]
            best_action = action
        end
    end

    return best_action, counts
end

function generate_dirichlet(rng, n, alpha)
    samples = Float32[]
    for i in 1:n
        push!(samples, sample_gamma(rng, alpha))
    end
    s = sum(samples)
    return samples ./ s
end

function sample_gamma(rng, alpha)
    if alpha < 1.0f0
        return sample_gamma(rng, alpha + 1.0f0) * (Float32(rand(rng))^(1.0f0 / alpha))
    end
    d = alpha - 1.0f0 / 3.0f0
    c = 1.0f0 / sqrt(9.0f0 * d)
    while true
        x = Float32(randn(rng))
        v = (1.0f0 + c * x)^3
        if v <= 0 return sample_gamma(rng, alpha) end
        u = Float32(rand(rng))
        if u < 1.0f0 - 0.0331 * x^4; return d * v; end
        if log(u) < 0.5 * x^2 + d * (1.0f0 - v + log(v)); return d * v; end
    end
end

function select_and_expand(mcts::MCTSSearch, root::MCTSNode)
    path = MCTSNode[]
    curr = root
    while true
        push!(path, curr)
        if is_terminal(curr.state)
            return curr, path
        end
        if isempty(curr.children)
            expand(mcts, curr)
            return curr, path 
        end
        action = select_puct(mcts, curr)
        curr = curr.children[action]
    end
end

function expand(mcts::MCTSSearch, node::MCTSNode)
    actions = legal_actions(node.state)
    if isempty(actions)
        return
    end

    logits, _ = predict(mcts.model, node.state)
    logits = vec(logits)

    mask = zeros(Float32, 6)
    for a in actions
        mask[a] = 1f0
    end

    logits .= logits .+ log.(mask .+ 1e-8)

    probs = softmax(logits)

    # solo probabilidades legales
    legal_probs = Float32[]
    for a in actions
        push!(legal_probs, probs[a])
    end

    s = sum(legal_probs)
    if s < 1e-8
        legal_probs .= 1f0 / length(legal_probs)
    else
        legal_probs ./= s
    end

    for (i, action) in enumerate(actions)
        s_next = transition(node.state, action)
        child_state = canonicalize(s_next)
        prior = legal_probs[i]
        
        if haskey(mcts.transposition_table, child_state.history_hash)
            (q_cached, n_cached) = mcts.transposition_table[child_state.history_hash]
            # We can initialize the child node with cached knowledge if applicable
            # But for a clean MCTS, we typically just expand and rely on select_puct's TT usage.
        end
        
        node.children[action] = MCTSNode(child_state, prior)
    end
end

function select_puct(mcts::MCTSSearch, node::MCTSNode)
    best_u = -Float32(Inf)
    best_a = 0
    n_parent = node.visits[]
    for (action, child) in node.children
        # Check Transposition Table for cached statistics
        q = child.value_sum[] / max(1, child.visits[])
        n = child.visits[]
        
        if haskey(mcts.transposition_table, child.state.history_hash)
            (q_cached, n_cached) = mcts.transposition_table[child.state.history_hash]
            # Use cached values to bias selection toward historically valuable branches
            u = q_cached + mcts.c_puct * child.prior * sqrt(n_parent) / (1 + n_cached)
        else
            u = q + mcts.c_puct * child.prior * sqrt(n_parent) / (1 + n)
        end
        
        if u > best_u
            best_u = u
            best_a = action
        end
    end
    return best_a
end

function backup(path::Vector{MCTSNode}, leaf_val::Float32, mcts::MCTSSearch)
    v = leaf_val
    for i in length(path):-1:1
        node = path[i]
        node.visits[] += 1
        node.value_sum[] += v
        
        # Update Transposition Table with aggregated value and visits
        q_avg = node.value_sum[] / node.visits[]
        n_total = node.visits[]
        mcts.transposition_table[node.state.history_hash] = (q_avg, n_total)
        
        v = -v
    end
end

end # module
