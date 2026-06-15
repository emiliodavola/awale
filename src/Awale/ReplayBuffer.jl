module ReplayBuffer

using ..State: GameState
using Random

export Experience, ReplayBuffer, push_experience!, sample_batch!

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

function push_experience!(rb::ReplayBuffer, exp::Experience)
    if length(rb.buffer) < rb.capacity
        push!(rb.buffer, exp)
    else
        rb.buffer[rb.cursor] = exp
        rb.cursor = (rb.cursor % rb.capacity) + 1
    end
end

function sample_batch!(rb::ReplayBuffer, batch_size::Int, rng=Random.default_rng())::Vector{Experience}
    if isempty(rb.buffer)
        return Experience[]
    end
    
    actual_batch_size = min(batch_size, length(rb.buffer))
    indices = sample(rng, 1:length(rb.buffer), actual_batch_size, replace=false)
    return [rb.buffer[i] for i in indices]
end

end # module
