using Test
using StaticArrays
using .Awale

@testset "env api" begin
    s = Awale.initial_state()
    actions = Awale.legal_actions(s)
    @test length(actions) == 6
    @test all(x -> 1 <= x <= 6, actions)

    # Test a simple move (P1 moves pit 1: has 4 seeds)
    s_next = Awale.transition(s, 1)

    @test s_next.board[1] == 0
    @test s_next.board[2] == 5
    @test s_next.board[5] == 5
    @test s_next.board[6] == 4
    @test s_next.board[7] == 4
    @test s_next.to_move == 2
    @test Awale.validate_invariants(s_next)

    # ---------------------------
    # FIX: GameState constructor
    # ---------------------------

    board_cap = SVector{12, UInt8}(
        UInt8[26,0,0,0,0,1, 2,3,4,4,4,4]
    )

    s_cap = Awale.GameState(
        board_cap,
        Int8(1),
        (UInt8(0), UInt8(0)),
        UInt64(0),
        Awale.GameConfig(),
        Set{UInt64}()
    )

    # Move pit 6 -> seed lands in pit 7. Pit 7 has 2, becomes 3. Capture!
    s_cap_next = Awale.transition(s_cap, 6)

    @test s_cap_next.captured[1] >= 3
    @test Awale.validate_invariants(s_cap_next)
end