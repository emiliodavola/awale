using Test
# Use absolute paths relative to this script
root = joinpath(@__DIR__, "..")
include(joinpath(root, "src/Awale/State.jl"))
include(joinpath(root, "src/Awale/Env.jl"))

function test_forced_feeding()
    println("Testing Forced Feeding...")
    cfg = State.GameConfig(forced_feeding=:require_feed)
    board = (4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0)
    s = State.GameState(board, Int8(1), (0, 0), 0, cfg)
    actions = Env.legal_actions(s)
    @test 6 in actions
    
    board_limited = (1, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0)
    s_limited = State.GameState(board_limited, Int8(1), (0, 0), 0, cfg)
    actions_limited = Env.legal_actions(s_limited)
    @test 6 in actions_limited
    @test !(1 in actions_limited)
end

function test_starvation_prevention()
    println("Testing Starvation Prevention...")
    cfg = State.GameConfig(starvation=:prevent_starvation)
    board = (4, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 0)
    s = State.GameState(board, Int8(1), (0, 0), 0, cfg)
    @test 6 in Env.legal_actions(s)
    
    board_danger = (0, 0, 0, 0, 10, 0, 1, 0, 0, 0, 0, 0)
    s_danger = State.GameState(board_danger, Int8(1), (0, 0), 0, cfg)
    actions_danger = Env.legal_actions(s_danger)
    @test !(5 in actions_danger)
end

try
    test_forced_feeding()
    test_starvation_prevention()
    println("All variant tests passed!")
catch e
    rethrow(e)
end
