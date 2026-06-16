using BenchmarkTools
include("src/Awale.jl")
using Awale.State
using Awale.MCTS
using Awale.Model
using Random
using StaticArrays

# Setup for microbenchmarking
function setup_benchmarks()
    config = Awale.GameConfig()
    s = Awale.initial_state(config)
    
    # Create a dummy model and MCTS for testing
    model = Awale.create_model()
    mcts = Awale.MCTSSearch(model, 1.4f0, Dict{UInt64, Tuple{Float32, Int64}}())
    
    # Create a node with children for testing
    root = Awale.MCTS.MCTSNode(s)
    for a in 1:6
        s_next = Awale.transition(s, a)
        child = Awale.MCTS.MCTSNode(s_next, 1.0f0)
        root.children[a] = child
        for ba in 1:2
            s_next_next = Awale.transition(s_next, ba)
            child_child = Awale.MCTS.MCTSNode(s_next_next, 1.0f0)
            child.children[ba] = child_child
        end
    end
    
    return (s, mcts, root)
end

function main()
    println("--- Microbenchmarks ---")
    (s, mcts, root) = setup_benchmarks()
    
    println("\n1. Encoding State (Optimization Check):")
    @btime Awale.Model.encode_state($s)
    
    println("\n2. Selection (PUCT):")
    @btime Awale.MCTS.select_puct($mcts, $root)
    
    println("\n3. Backup (Memory & TT):")
    path = [root, root.children[1], root.children[1].children[1]]
    @btime Awale.MCTS.backup($path, 1.0f0, $mcts)
end

main()
