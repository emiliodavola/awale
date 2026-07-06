module Training

using ..State: GameState, initial_state, GameConfig, canonicalize
using ..Env: is_terminal, transition, reward
using ..Model: predict_raw, encode_state
using ..MCTS: MCTSSearch, search_with_stats
using ..ReplayBuffers: Experience, ReplayBuffer, push_experience!, sample_batch
using Flux
using Random

export play_game, collect_selfplay_data, train_step, run_training_iteration, backfill_value_targets

function temperature_for_turn(turns_played::Int, temperature_moves::Int)::Float32
    return turns_played < temperature_moves ? 1.0f0 : 0.0f0
end

function sample_action_from_policy(pi_target::Vector{Float32}, rng, temperature::Float32)::Int
    if temperature <= 0.0f0
        return argmax(pi_target)
    end

    weights = copy(pi_target)
    total = sum(weights)
    if total <= 0.0f0
        return argmax(weights)
    end

    weights ./= total
    cumulative = 0.0f0
    threshold = rand(rng, Float32)
    for action in eachindex(weights)
        cumulative += weights[action]
        if threshold <= cumulative && weights[action] > 0.0f0
            return action
        end
    end

    return argmax(weights)
end

function backfill_value_targets(samples::Vector{Tuple{GameState, Vector{Float32}, Float32}}, terminal_reward::Float32)
    backfilled = copy(samples)
    current_value = -terminal_reward

    for i in length(backfilled):-1:1
        state, pi_target, _ = backfilled[i]
        backfilled[i] = (state, pi_target, current_value)
        current_value = -current_value
    end

    return backfilled
end

# Plays a single self-play game and returns (state, pi_target, v_target) tuples.
function collect_selfplay_data(
    mcts::MCTSSearch,
    config::GameConfig=GameConfig(),
    sims_per_move::Int=100,
    temperature_moves::Int=20,
    rng=Random.default_rng(),
)
    state = initial_state(config)
    samples = Tuple{GameState, Vector{Float32}, Float32}[]
    turns_played = 0
    max_turns = 1000

    while !is_terminal(state) && turns_played < max_turns
        _, pi_target = search_with_stats(mcts, state, sims_per_move, rng; add_root_noise=true)
        push!(samples, (canonicalize(state), pi_target, 0.0f0))

        temperature = temperature_for_turn(turns_played, temperature_moves)
        action = sample_action_from_policy(pi_target, rng, temperature)
        state = transition(state, action)
        turns_played += 1
    end

    return backfill_value_targets(samples, reward(state))
end

function train_step(model, optimizer, states, target_pis, target_vs)
    X = hcat([vec(encode_state(canonicalize(state))) for state in states]...)
    Y_pi = hcat(target_pis...)
    Y_v = reshape(target_vs, 1, :)

    function loss_fn(current_model)
        logits, values = predict_raw(current_model, X)
        log_probs = Flux.logsoftmax(logits, dims=1)
        policy_loss = -sum(Y_pi .* log_probs) / size(X, 2)
        value_loss = Flux.mse(values, Y_v)
        return policy_loss + value_loss
    end

    grads = Flux.gradient(loss_fn, model)[1]
    Flux.update!(optimizer, model, grads)
    return loss_fn(model)
end

function run_training_iteration(
    mcts::MCTSSearch,
    optimizer,
    model,
    replay_buffer::ReplayBuffer;
    n_games::Int=5,
    sims::Int=100,
    batch_size::Int=64,
    updates_per_iteration::Int=16,
    replay_recent_fraction::Float64=0.5,
    replay_recent_window::Int=4096,
    temperature_moves::Int=20,
    rng=Random.default_rng(),
)
    for game_idx in 1:n_games
        print("  Game $game_idx/$n_games... ")
        game_data = collect_selfplay_data(mcts, GameConfig(), sims, temperature_moves, rng)
        for (state, pi_target, value_target) in game_data
            push_experience!(replay_buffer, Experience(state, pi_target, value_target))
        end
        println("Done.")
    end

    losses = Float32[]
    for _ in 1:updates_per_iteration
        batch = sample_batch(
            replay_buffer,
            batch_size,
            rng;
            recent_fraction=replay_recent_fraction,
            recent_window=replay_recent_window,
        )
        isempty(batch) && break

        states = [experience.state for experience in batch]
        pi_targets = [experience.pi_target for experience in batch]
        value_targets = Float32[experience.z_target for experience in batch]
        push!(losses, train_step(model, optimizer, states, pi_targets, value_targets))
    end

    return isempty(losses) ? 0.0f0 : sum(losses) / length(losses)
end

function play_game(mcts, config=GameConfig(), sims=100, temperature_moves=20, rng=Random.default_rng())
    return collect_selfplay_data(mcts, config, sims, temperature_moves, rng)
end

end # module
