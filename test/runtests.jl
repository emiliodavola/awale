using Test

# Load project code directly for development-mode tests
include("../src/Awale.jl")
using .Awale

include("test_state_model.jl")
include("test_model_mode_contract.jl")
include("test_env_api.jl")
include("test_invariants.jl")
include("test_variants.jl")
include("test_training_pipeline.jl")
include("test_publication_flow.jl")
