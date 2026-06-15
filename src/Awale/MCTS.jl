module MCTS

using ..State: GameState, canonicalize
using ..Env: legal_actions, transition, is_terminal, reward
using ..Model: AwaleNet, predict

export MCTSSearch, search

struct MCTSNode
    state::GameState
    prior::Float32 # Prior probability from NN (for root, it is 1.0 usually)
    visits::Ref{Int64}
    value_sum::Ref{Float32}
    children::Dict{Int, MCTSNode}
end

function MCTSNode(s::GameState, prior::Float32=1.0f0)
    return MCTSNode(s, prior, Ref(0), Ref(0.0f0), Dict())
end

struct MCTSSearch
    model::AwaleNet
    c_puct::Float32
end

function search(mcts::MCTSSearch, root_state::GameState, num_sims::Int)
    root = MCTSNode(canonicalize(root_state))
    
    for _ in 1:num_sims
        # Selection & Expansion
        leaf, path = select_and_expand(mcts, root)
        
        # Evaluation (if not terminal)
        val = if is_terminal(leaf.state)
            reward(leaf.state) # from current player perspective
        else
            # Use NN for value
            (_, v) = predict(mcts.model, leaf.state)
            v
        end
        
        # Backup
        backup(path, val)
    end
    
    # Return the best action (most visited)
    best_action = 0
    max_visits = -1
    for (action, node) in root.children
        if node.visits[] > max_visits
            max_visits = node.visits[]
            best_action = action
        end
    end
    return best_action
end

function select_and_expand(mcts::MCTSSearch, root::MCTSNode)
    path = MCTSNode[]
    curr = root
    
    while true
        push!(path, curr)
        if is_terminal(curr.state)
            return curr, path
        end
        
        # If not expanded, expand now
        if isempty(curr.children)
            expand(mcts, curr)
            return curr, path # The leaf we just expanded (or one of its children if we want a deeper search)
            # Actually AlphaZero expands and then evaluates the leaf. 
            # So returning the leaf before evaluating is correct.
        end
        
        # Selection via PUCT
        action = select_puct(mcts, curr)
        curr = curr.children[action]
    end
end

function expand(mcts::MCTSSearch, node::MCTSNode)
    actions = legal_actions(node.state)
    if isempty(actions)
        return
    end
    
    # Get priors from NN
    logits, _ = predict(mcts.model, node.state)
    # Softmax for probabilities
    probs = exp.(logits) / sum(exp.(logits))
    
    # Only create children for legal actions and renormalize
    legal_probs = Float32[]
    for a in 1:6
        if a in actions
            push!(legal_probs, probs[a])
        end
    end
    sum_lp = sum(legal_probs)
    legal_probs = legal_probs ./ sum_lp
    
    # Create nodes for legal actions
    for (i, action) in enumerate(actions)
        s_next = transition(node.state, action)
        node.children[action] = MCTSNode(canonicalize(s_next), legal_probs[i])
    end
end

function select_puct(mcts::MCTSSearch, node::MCTSNode)
    best_u = -Float32(Inf)
    best_a = 0
    
    n_parent = node.visits[]
    
    for (action, child) in node.children
        q = child.value_sum[] / max(1, child.visits[])
        u = q + mcts.c_puct * child.prior * sqrt(n_parent) / (1 + child.visits[])
        if u > best_u
            best_u = u
            best_a = action
        end
    end
    return best_a
end

function backup(path::Vector{MCTSNode}, leaf_val::Float32)
    # path is [root, ..., leaf]
    # Backup inverts perspective: v_parent = -v_child
    v = leaf_val
    for i in length(path):-1:1
        node = path[i]
        node.visits[] += 1
        node.value_sum[] += v
        v = -v # invert for next level up
    end
end

end # module
