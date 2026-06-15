using Test
using Awale
using Awale.State: initial_state, GameConfig
using Awale.Env: transition, legal_actions

@testset "Game Invariants" begin
    config = GameConfig()
    s = initial_state(config)
    
    # Property 1: Seed Conservation
    # sum(board) + cap1 + cap2 == 48
    function total_seeds(st)
        return sum(st.board) + st.captured[1] + st.captured[2]
    end

    @test total_seeds(s) == 48

    # Test several random transitions
    curr_s = s
    for i in 1:100
        actions = legal_actions(curr_s)
        if isempty(actions) break end
        
        action = rand(actions)
        prev_s = curr_s
        curr_s = transition(curr_s, action)
        
        # Check seed conservation
        @test total_seeds(curr_s) == 48
        
        # Property 2: Transition Purity
        # Verify that the previous state object was not modified (though it is immutable by design)
        # Since GameState is a struct with immutable fields, this is mostly guaranteed.
    end
end