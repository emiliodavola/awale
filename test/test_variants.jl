using Test
using StaticArrays: SVector
using .Awale

function variant_state(board::NTuple{12,Int}, config::Awale.GameConfig; to_move::Int8=Int8(1), captured=(UInt8(0), UInt8(0)))
    board_vec = SVector{12,UInt8}(UInt8.(board))
    return Awale.GameState(board_vec, to_move, captured, UInt64(0), config, Set{UInt64}())
end

@testset "rule variants" begin
    @testset "forced feeding requires a feeding move when one exists" begin
        @test Awale.GameConfig().forced_feeding == :require_feed
        cfg = Awale.GameConfig()
        state = variant_state((4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0), cfg)
        actions = Awale.legal_actions(state)
        @test 6 in actions

        limited_state = variant_state((1, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0), cfg)
        limited_actions = Awale.legal_actions(limited_state)
        @test 6 in limited_actions
        @test !(1 in limited_actions)
    end

    @testset "starvation prevention filters starving moves when alternatives exist" begin
        cfg = Awale.GameConfig(starvation=:prevent_starvation)
        safe_state = variant_state((4, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 0), cfg)
        @test 6 in Awale.legal_actions(safe_state)

        filtered_state = variant_state((1, 0, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0), cfg)
        filtered_actions = Awale.legal_actions(filtered_state)
        @test !(1 in filtered_actions)
        @test 3 in filtered_actions
        @test 4 in filtered_actions
    end
end
