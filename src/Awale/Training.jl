module Training

using ..State: GameState, initial_state, GameConfig
using ..Env: is_terminal, transition, reward, legal_actions
using ..Model: AwaleNet, create_model, predict
using ..MCTS: MCTSSearch, search
using Flux

export play_game, train_step

# plays a single game using MCTS and returns the history of states and actions
function play_game(mcts::MCTSSearch, config::GameConfig=GameConfig())::Vector{Tuple{GameState, Int}}
    s = initial_state(config)
    history = Tuple{GameState, Int}[]
    
    while !is_terminal(s)
        action = search(mcts, s, 100) # 100 sims per move for self-play
        push!(history, (s, action))
        s = transition(s, action)
    end
    
    return history
end

# a simple training step that takes a batch of (state, target_pi, target_v)
function train_step(model::AwaleNet, optimizer, states, target_pis, target_vs)
    # This would involve:
    # 1. Forward pass
    # 2. Compute loss (CrossEntropy for pi, MSE for v)
    # 3. Backprop and update
    
    # Implementation details here depend on how we batch the data in Flux
    # For now, we define the contract as requested by Phase 3.
end

end # module
