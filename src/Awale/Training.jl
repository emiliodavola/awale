"""
    Training

Self-play data generation, neural network training steps, and orchestration
of training iterations with experience replay.
"""
module Training

using ..State: GameState, initial_state, GameConfig, canonicalize
using ..Env: is_terminal, transition, reward
using ..Model: predict_raw, encode_state
using ..MCTS: MCTSSearch, search_with_stats
using ..ReplayBuffers: Experience, ReplayBuffer, push_experience!, sample_batch
using Flux
using Random

export play_game, collect_selfplay_data, train_step, run_training_iteration, backfill_value_targets

"""
    temperature_for_turn(turns_played, temperature_moves) -> Float32

Return 1.0 for early turns (within `temperature_moves`) and 0.0 after,
switching from exploratory sampling to argmax.
"""
function temperature_for_turn(turns_played::Int, temperature_moves::Int)::Float32
    return turns_played < temperature_moves ? 1.0f0 : 0.0f0
end

"""
    sample_action_from_policy(pi_target, rng, temperature) -> Int

Sample an action from the target policy. If temperature <= 0, picks argmax.
Otherwise, samples proportionally from the normalized policy weights.
"""
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

"""
    backfill_value_targets(samples, terminal_reward) -> Vector{Tuple}

Backpropagate terminal rewards through a game trajectory with sign inversion.
Returns a copy with corrected value targets (z_target) for each position.
"""
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

"""
    collect_selfplay_data(mcts, config, sims_per_move, temperature_moves, rng; max_turns) -> Vector{Tuple}

Play a single self-play game and return (state, pi_target, v_target) tuples
with backfilled value targets. Sampling uses temperature annealing.
All runtime parameters must be provided explicitly by the caller.
"""
function collect_selfplay_data(
    mcts::MCTSSearch,
    config::GameConfig,
    sims_per_move::Int,
    temperature_moves::Int,
    rng;
    max_turns::Int,
)
    state = initial_state(config)
    samples = Tuple{GameState, Vector{Float32}, Float32}[]
    turns_played = 0

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

"""
    train_step(model, optimizer, states, target_pis, target_vs) -> Float32

Single gradient update: compute policy + value loss, backpropagate, and
apply the optimizer. Returns the combined loss value.
"""
function train_step(model, optimizer, states, target_pis, target_vs)
    X = hcat([vec(encode_state(canonicalize(state))) for state in states]...)
    Y_pi = hcat(target_pis...)
    Y_v = reshape(target_vs, 1, :)

    function loss_fn(current_model)
        logits, values = predict_raw(current_model, X)
        log_probs = Flux.logsoftmax(logits, dims=1)
        policy_loss = -sum(Y_pi .* log_probs) / size(X, 2)
        value_loss = Flux.mse(values, Y_v)
        return policy_loss, value_loss, log_probs
    end

    grads = Flux.gradient(m -> begin
        pl, vl, _ = loss_fn(m)
        return pl + vl
    end, model)[1]
    Flux.update!(optimizer, model, grads)

    # Compute diagnostics on the updated model
    after_logits, after_values = predict_raw(model, X)
    after_log_probs = Flux.logsoftmax(after_logits, dims=1)
    after_probs = exp.(after_log_probs)
    policy_loss = -sum(Y_pi .* after_log_probs) / size(X, 2)
    value_loss = Flux.mse(after_values, Y_v)
    combined_loss = policy_loss + value_loss

    # Gradient norm — recursively flatten Flux NamedTuple params
    function flat_norm(x)
        if x isa Number
            return Float32(abs2(x))
        elseif x isa AbstractArray
            return Float32(sum(abs2, x))
        elseif x isa Tuple || x isa NamedTuple
            total = 0.0f0
            for field in x
                total += flat_norm(field)
            end
            return total
        elseif x === nothing
            return 0.0f0
        else
            try
                return flat_norm(Flux.params(x))
            catch
                return 0.0f0
            end
        end
    end
    grad_norm = sqrt(flat_norm(grads))

    # Predicted policy entropy: H(p_pred) = -sum(p_pred * log p_pred)
    pred_entropy = -sum(after_probs .* after_log_probs) / size(X, 2)

    # Target policy entropy: H(pi_target)
    safe_target = clamp.(Y_pi, 1.0f-10, 1.0f0)
    target_entropy = -sum(Y_pi .* log.(safe_target)) / size(X, 2)

    return (
        loss=combined_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        grad_norm=grad_norm,
        pred_entropy=pred_entropy,
        target_entropy=target_entropy,
    )
end

"""
    run_training_iteration(mcts, optimizer, model, replay_buffer; kwargs) -> Float32

Run one training iteration: generate self-play games, collect experiences into
the replay buffer, then perform gradient updates on sampled minibatches.
Returns the average combined loss across all updates.
"""
function run_training_iteration(
    mcts::MCTSSearch,
    optimizer,
    model,
    replay_buffer::ReplayBuffer;
    n_games::Int,
    sims::Int,
    batch_size::Int,
    updates_per_iteration::Int,
    replay_recent_fraction::Float64,
    replay_recent_window::Int,
    temperature_moves::Int,
    rng,
    max_turns::Int,
)
    recent_pct = Int(round(replay_recent_fraction * 100))
    history_pct = 100 - recent_pct

    total_positions = 0
    total_game_length = 0

    for game_idx in 1:n_games
        print("\r  Self-play: $game_idx/$n_games")
        flush(stdout)
        game_data = collect_selfplay_data(mcts, GameConfig(), sims, temperature_moves, rng; max_turns=max_turns)
        total_positions += length(game_data)
        total_game_length += length(game_data)
        for (state, pi_target, value_target) in game_data
            push_experience!(replay_buffer, Experience(state, pi_target, value_target))
        end
    end
    println("\r  Self-play: $n_games/$n_games | updates: $updates_per_iteration | replay mix: $(recent_pct)% recent / $(history_pct)% history")

    losses = Float32[]
    policy_losses = Float32[]
    value_losses = Float32[]
    grad_norms = Float32[]
    pred_entropies = Float32[]
    target_entropies = Float32[]
    total_samples = 0

    for _ in 1:updates_per_iteration
        batch = sample_batch(
            replay_buffer,
            batch_size,
            rng;
            recent_fraction=replay_recent_fraction,
            recent_window=replay_recent_window,
        )
        isempty(batch) && break

        total_samples += length(batch)
        states = [experience.state for experience in batch]
        pi_targets = [experience.pi_target for experience in batch]
        value_targets = Float32[experience.z_target for experience in batch]

        step_result = train_step(model, optimizer, states, pi_targets, value_targets)
        push!(losses, step_result.loss)
        push!(policy_losses, step_result.policy_loss)
        push!(value_losses, step_result.value_loss)
        push!(grad_norms, step_result.grad_norm)
        push!(pred_entropies, step_result.pred_entropy)
        push!(target_entropies, step_result.target_entropy)
    end

    avg_loss = isempty(losses) ? 0.0f0 : sum(losses) / length(losses)

    if !isempty(policy_losses)
        replay_capacity = replay_buffer.capacity
        replay_fill = length(replay_buffer)
        replay_pct = round(replay_fill / replay_capacity * 100, digits=1)
        avg_game_len = total_positions / max(1, n_games)
        avg_policy = sum(policy_losses) / length(policy_losses)
        avg_value = sum(value_losses) / length(value_losses)
        avg_grad = sum(grad_norms) / length(grad_norms)
        avg_pred_ent = sum(pred_entropies) / length(pred_entropies)
        avg_target_ent = sum(target_entropies) / length(target_entropies)

        println("  ── Diagnostics ──────────────────────────────────")
        println("    Positions generated      : $total_positions")
        println("    Avg game length          : $(round(avg_game_len, digits=1))")
        println("    Samples consumed         : $total_samples")
        println("    Replay coverage          : $replay_pct %")
        println("    Avg policy loss          : $(round(avg_policy, digits=4))")
        println("    Avg value loss           : $(round(avg_value, digits=4))")
        println("    Avg gradient norm        : $(round(avg_grad, digits=4))")
        println("    Target policy entropy    : $(round(avg_target_ent, digits=4))")
        println("    Predicted policy entropy : $(round(avg_pred_ent, digits=4))")
        println("  ─────────────────────────────────────────────────")
    end

    return avg_loss
end

"""
    play_game(mcts, config, sims, temperature_moves, rng; max_turns) -> Vector{Tuple}

Convenience wrapper for `collect_selfplay_data`.
"""
function play_game(mcts, config, sims, temperature_moves, rng; max_turns::Int)
    return collect_selfplay_data(mcts, config, sims, temperature_moves, rng; max_turns=max_turns)
end

end # module