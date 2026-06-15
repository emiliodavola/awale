<<<<<<< HEAD
using Test
using .Awale

@testset "env api" begin
    s = Awale.initial_state()
    actions = Awale.legal_actions(s)
    @test length(actions) == 6
    @test all(x->1<=x<=6, actions)
    
    # Test a simple move (P1 moves pit 1: has 4 seeds)
    # Sowing fills pits 2, 3, 4, 5. Pit 6 and 7 remain unchanged.
    s_next = Awale.transition(s, 1)
    @test s_next.board[1] == 0
    @test s_next.board[2] == 5
    @test s_next.board[5] == 5
    @test s_next.board[6] == 4 # unchanged
    @test s_next.board[7] == 4 # unchanged
    @test s_next.to_move == 2
    @test Awale.validate_invariants(s_next)

    # Test capture: Create a state with total 48 seeds, where P1 can capture
    # P1 pits: [0,0,0,0,0,1] (sum=1)
    # P2 pits: [2,3,4,4,4,4] (sum=21). Total = 22.
    # We need 26 more seeds in P1's captured or just add to board.
    # Let\'s put them all in P1's pit 1 for simplicity? No, then it is not a "capture test" setup.
    # Let\'s make s_cap have 48 seeds total:
    # board = [26,0,0,0,0,1,  2,3,4,4,4,4] (sum = 26+1+21 = 48)
    board_cap = NTuple{12,UInt8}([26,0,0,0,0,1,  2,3,4,4,4,4])
    s_cap = Awale.GameState(board_cap, Int8(1), (0,0), 0, Awale.GameConfig())
    
    # Move pit 6 -> seed lands in pit 7. Pit 7 has 2, becomes 3. Capture!
    s_cap_next = Awale.transition(s_cap, 6)
    @test s_cap_next.captured[1] >= 3
    @test Awale.validate_invariants(s_cap_next)
end
=======
using Test
using .Awale

@testset "env api" begin
    s = Awale.initial_state()
    actions = Awale.legal_actions(s)
    @test length(actions) == 6
    @test all(x->1<=x<=6, actions)
    
    # Test a simple move (P1 moves pit 1: has 4 seeds)
    # Sowing fills pits 2, 3, 4, 5. Pit 6 and 7 remain unchanged.
    s_next = Awale.transition(s, 1)
    @test s_next.board[1] == 0
    @test s_next.board[2] == 5
    @test s_next.board[5] == 5
    @test s_next.board[6] == 4 # unchanged
    @test s_next.board[7] == 4 # unchanged
    @test s_next.to_move == 2
    @test Awale.validate_invariants(s_next)

    # Test capture: Create a state with total 48 seeds, where P1 can capture
    # P1 pits: [0,0,0,0,0,1] (sum=1)
    # P2 pits: [2,3,4,4,4,4] (sum=21). Total = 22.
    # We need 26 more seeds in P1's captured or just add to board.
    # Let\'s put them all in P1's pit 1 for simplicity? No, then it is not a "capture test" setup.
    # Let\'s make s_cap have 48 seeds total:
    # board = [26,0,0,0,0,1,  2,3,4,4,4,4] (sum = 26+1+21 = 48)
    board_cap = NTuple{12,UInt8}([26,0,0,0,0,1,  2,3,4,4,4,4])
    s_cap = Awale.GameState(board_cap, Int8(1), (0,0), 0, Awale.GameConfig())
    
    # Move pit 6 -> seed lands in pit 7. Pit 7 has 2, becomes 3. Capture!
    s_cap_next = Awale.transition(s_cap, 6)
    @test s_cap_next.captured[1] >= 3
    @test Awale.validate_invariants(s_cap_next)
end
>>>>>>> origin/dev
