<<<<<<< HEAD
using Test

# Load project code directly for development-mode tests
include("../src/Awale.jl")
using .Awale

include("test_state_model.jl")
include("test_env_api.jl")
=======
using Test

# Load project code directly for development-mode tests
include("../src/Awale.jl")
using .Awale

include("test_state_model.jl")
include("test_env_api.jl")
include("test_invariants.jl")
>>>>>>> origin/dev
