module ReplayBuffers

using ..State: GameState
using Random

export Experience, ReplayBuffer, push_experience!, chronological_indices, sample_batch

struct Experience
    state::GameState
    pi_target::Vector{Float32}
    z_target::Float32
end

mutable struct ReplayBuffer
    capacity::Int
    buffer::Vector{Experience}
    cursor::Int
end

function ReplayBuffer(capacity::Int)
    return ReplayBuffer(capacity, Experience[], 1)
end

Base.length(rb::ReplayBuffer) = length(rb.buffer)

function push_experience!(rb::ReplayBuffer, exp::Experience)
    if length(rb) < rb.capacity
        push!(rb.buffer, exp)
    else
        rb.buffer[rb.cursor] = exp
        rb.cursor = (rb.cursor % rb.capacity) + 1
    end
end

function chronological_indices(rb::ReplayBuffer)::Vector{Int}
    if length(rb) < rb.capacity
        return collect(1:length(rb))
    end

    return vcat(collect(rb.cursor:length(rb)), collect(1:(rb.cursor - 1)))
end

function sample_without_replacement(pool::Vector{Int}, count::Int, rng)
    count <= 0 && return Int[]
    actual_count = min(count, length(pool))
    actual_count == 0 && return Int[]
    return pool[randperm(rng, length(pool))[1:actual_count]]
end

function sample_batch(
    rb::ReplayBuffer,
    batch_size::Int,
    rng=Random.default_rng();
    recent_fraction::Float64=0.0,
    recent_window::Int=length(rb),
)::Vector{Experience}
    0.0 <= recent_fraction <= 1.0 || throw(ArgumentError("recent_fraction must be between 0 and 1"))
    recent_window >= 0 || throw(ArgumentError("recent_window must be >= 0"))
    recent_fraction == 0.0 || recent_window > 0 || throw(ArgumentError("recent_window must be > 0 when recent_fraction > 0"))

    if isempty(rb.buffer)
        return Experience[]
    end

    actual_batch_size = min(batch_size, length(rb))
    if recent_fraction == 0.0
        indices = randperm(rng, length(rb))[1:actual_batch_size]
        return [rb.buffer[i] for i in indices]
    end

    ordered = chronological_indices(rb)
    tail_size = min(recent_window, length(ordered))
    recent_pool = ordered[(end - tail_size + 1):end]
    history_pool = ordered[1:(end - tail_size)]
    recent_count = min(Int(ceil(actual_batch_size * recent_fraction)), length(recent_pool), actual_batch_size)

    chosen_recent = sample_without_replacement(recent_pool, recent_count, rng)
    chosen_history = sample_without_replacement(history_pool, actual_batch_size - length(chosen_recent), rng)

    if length(chosen_recent) + length(chosen_history) < actual_batch_size
        remaining_pool = [idx for idx in recent_pool if idx ∉ chosen_recent]
        append!(chosen_history, sample_without_replacement(remaining_pool, actual_batch_size - length(chosen_recent) - length(chosen_history), rng))
    end

    indices = vcat(chosen_recent, chosen_history)
    return [rb.buffer[i] for i in indices]
end

end # module
