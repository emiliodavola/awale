<<<<<<< HEAD
module Awale

# Awale: specification-first Awale RL project

include("Awale/Utils.jl")
include("Awale/State.jl")
include("Awale/Env.jl")
include("Awale/Model.jl")
include("Awale/MCTS.jl")
include("Awale/Training.jl")

using .Utils
using .State
using .Env
using .Model
using .MCTS
using .Training

export GameConfig, GameState, initial_state, canonicalize, serialize_state, deserialize_state, hash_state, validate_invariants, legal_actions, transition, AwaleNet, create_model, predict, MCTSSearch, search, play_game, train_step

end # module
=======
module Awale

# Awale: specification-first Awale RL project

include("Awale/Utils.jl")
include("Awale/State.jl")
include("Awale/Env.jl")
include("Awale/Model.jl")
include("Awale/MCTS.jl")
include("Awale/Training.jl")
include("Awale/Evaluation.jl")

using .Utils
using .State
using .Env
using .Model
using .MCTS
using .Training
using .Evaluation

export *

end # module
>>>>>>> origin/dev
