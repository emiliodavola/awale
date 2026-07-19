using Test
using StaticArrays
using .Awale

function terminal_state(board::NTuple{12,Int}; to_move::Int8=Int8(1), captured=(UInt8(0), UInt8(0)), config::Awale.GameConfig=Awale.GameConfig(), history_hash::UInt64=UInt64(0), history_hashes::Set{UInt64}=Set{UInt64}())
    board_vec = SVector{12,UInt8}(UInt8.(board))
    return Awale.GameState(board_vec, to_move, captured, history_hash, config, history_hashes)
end

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

    board_p2_cap = SVector{12, UInt8}(
        UInt8[1,1,0,0,0,0, 0,0,0,0,0,2]
    )

    s_p2_cap = Awale.GameState(
        board_p2_cap,
        Int8(2),
        (UInt8(22), UInt8(22)),
        UInt64(0),
        Awale.GameConfig(),
        Set{UInt64}()
    )

    s_p2_next = Awale.transition(s_p2_cap, 6)

    @test s_p2_next.captured[2] == 26
    @test s_p2_next.board[1] == 0
    @test s_p2_next.board[2] == 0
    @test Awale.validate_invariants(s_p2_next)

    @testset "terminal resolution rules" begin
        win_state = terminal_state((13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0); to_move=Int8(2), captured=(UInt8(10), UInt8(25)))
        @test Awale.is_terminal(win_state)
        @test Awale.reward(win_state) == 1.0f0
        @test Awale.Evaluation.result_from_terminal_state(win_state) == -1

        draw_state = terminal_state((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0); to_move=Int8(1), captured=(UInt8(24), UInt8(24)))
        @test Awale.is_terminal(draw_state)
        @test Awale.reward(draw_state) == 0.0f0
        @test Awale.Evaluation.result_from_terminal_state(draw_state) == 0

        no_feed_state = terminal_state((0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0); to_move=Int8(1), captured=(UInt8(20), UInt8(20)))
        @test Awale.is_terminal(no_feed_state)
        @test Awale.reward(no_feed_state) == -1.0f0
        @test Awale.Evaluation.result_from_terminal_state(no_feed_state) == -1

        repeated_draw = terminal_state((4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4); config=Awale.GameConfig(repetition=:draw_on_repeat), history_hash=UInt64(7), history_hashes=Set([UInt64(7)]))
        repeated_score = terminal_state((3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0); captured=(UInt8(20), UInt8(25)), config=Awale.GameConfig(repetition=:score_diff), history_hash=UInt64(7), history_hashes=Set([UInt64(7)]))
        repeated_revert = terminal_state((4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4); config=Awale.GameConfig(repetition=:revert), history_hash=UInt64(7), history_hashes=Set([UInt64(7)]))

        @test Awale.is_terminal(repeated_draw)
        @test Awale.reward(repeated_draw) == 0.0f0
        @test Awale.Evaluation.result_from_terminal_state(repeated_draw) == 0

        @test Awale.is_terminal(repeated_score)
        @test Awale.reward(repeated_score) == -1.0f0
        @test Awale.Evaluation.result_from_terminal_state(repeated_score) == -1

        @test !Awale.is_terminal(repeated_revert)
    end

    @testset "evaluation and arena result conversion stays aligned" begin
        no_feed_state = terminal_state((0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0); to_move=Int8(1), captured=(UInt8(20), UInt8(20)))
        @test Awale.Evaluation.result_from_terminal_state(no_feed_state) == -1
        @test Awale.reward(no_feed_state) == -1.0f0
    end

    @testset "match cutoff is reported explicitly" begin
        outcome = Awale.Evaluation.play_match_from_state(
            Awale.initial_state(),
            Awale.Evaluation.RandomAgent(),
            Awale.Evaluation.RandomAgent();
            max_turns=0,
        )

        @test outcome.result == 0
        @test outcome.turns_played == 0
        @test outcome.cutoff
    end

    @testset "terminal matches stay terminal" begin
        terminal = terminal_state((13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0); to_move=Int8(2), captured=(UInt8(10), UInt8(25)))
        outcome = Awale.Evaluation.play_match_from_state(
            terminal,
            Awale.Evaluation.RandomAgent(),
            Awale.Evaluation.RandomAgent();
            max_turns=0,
        )

        @test outcome.result == -1
        @test outcome.turns_played == 0
        @test !outcome.cutoff
    end
end
