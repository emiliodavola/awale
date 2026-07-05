using Test
using Flux
using Random
using .Awale

struct DummyModel end
struct Prior1Model end
struct Prior6Model end

function Awale.Model.predict(::DummyModel, ::Awale.GameState)
    return zeros(Float32, 6), 0.0f0
end

function Awale.Model.predict(::Prior1Model, ::Awale.GameState)
    return Float32[10, -10, -10, -10, -10, -10], 0.0f0
end

function Awale.Model.predict(::Prior6Model, ::Awale.GameState)
    return Float32[-10, -10, -10, -10, -10, 10], 0.0f0
end

@testset "training pipeline" begin
    @testset "value target backfill uses pre-terminal perspective" begin
        samples = [
            (Awale.initial_state(), fill(1.0f0 / 6.0f0, 6), 0.0f0),
            (Awale.initial_state(), fill(1.0f0 / 6.0f0, 6), 0.0f0),
        ]

        backfilled = Awale.backfill_value_targets(samples, -1.0f0)

        @test backfilled[2][3] == 1.0f0
        @test backfilled[1][3] == -1.0f0
    end

    @testset "policy head keeps unconstrained logits" begin
        model = Awale.create_model()
        @test model.policy.layers[end].σ === identity
    end

    @testset "search clears stale transposition state and supports deterministic eval mode" begin
        s = Awale.initial_state()
        rng = MersenneTwister(7)
        mcts = Awale.MCTSSearch(DummyModel(), 1.5f0, Dict{UInt64, Tuple{Float32, Int64}}(0xdeadbeef => (99.0f0, 99)))

        action_a, counts_a = Awale.search_with_stats(mcts, s, 8, rng; add_root_noise=false)

        @test !haskey(mcts.transposition_table, 0xdeadbeef)

        rng = MersenneTwister(7)
        action_b, counts_b = Awale.search_with_stats(mcts, s, 8, rng; add_root_noise=false)

        @test action_a == action_b
        @test counts_a == counts_b
    end

    @testset "policy-only and one-simulation search honor policy priors" begin
        s = Awale.initial_state()
        rng = MersenneTwister(3)

        action_1_zero, policy_1_zero = Awale.search_with_stats(Awale.MCTSSearch(Prior1Model(), 1.5f0, Dict{UInt64, Tuple{Float32, Int64}}()), s, 0, rng; add_root_noise=false)
        rng = MersenneTwister(3)
        action_6_zero, policy_6_zero = Awale.search_with_stats(Awale.MCTSSearch(Prior6Model(), 1.5f0, Dict{UInt64, Tuple{Float32, Int64}}()), s, 0, rng; add_root_noise=false)
        rng = MersenneTwister(3)
        action_1, counts_1 = Awale.search_with_stats(Awale.MCTSSearch(Prior1Model(), 1.5f0, Dict{UInt64, Tuple{Float32, Int64}}()), s, 1, rng; add_root_noise=false)
        rng = MersenneTwister(3)
        action_6, counts_6 = Awale.search_with_stats(Awale.MCTSSearch(Prior6Model(), 1.5f0, Dict{UInt64, Tuple{Float32, Int64}}()), s, 1, rng; add_root_noise=false)

        @test action_1_zero == 1
        @test policy_1_zero[1] > 0.99f0
        @test action_6_zero == 6
        @test policy_6_zero[6] > 0.99f0
        @test action_1 == 1
        @test counts_1[1] == 1.0f0
        @test action_6 == 6
        @test counts_6[6] == 1.0f0
    end

    @testset "transposition keys distinguish repetition history" begin
        s = Awale.initial_state()
        repeated = Awale.GameState(s.board, s.to_move, s.captured, s.history_hash, s.config, Set([s.history_hash]))

        @test s.history_hash == repeated.history_hash
        @test Awale.MCTS.transposition_key(s) != Awale.MCTS.transposition_key(repeated)
    end

    @testset "puct compares values from the parent perspective" begin
        root = Awale.MCTS.MCTSNode(Awale.initial_state())
        root.visits[] = 10

        child_bad_for_parent = Awale.MCTS.MCTSNode(Awale.initial_state(), 0.5f0)
        child_bad_for_parent.visits[] = 10
        child_bad_for_parent.value_sum[] = 10.0f0

        child_good_for_parent = Awale.MCTS.MCTSNode(Awale.initial_state(), 0.5f0)
        child_good_for_parent.visits[] = 10
        child_good_for_parent.value_sum[] = -10.0f0

        root.children[1] = child_bad_for_parent
        root.children[2] = child_good_for_parent

        mcts = Awale.MCTSSearch(DummyModel(), 1.5f0, Dict{UInt64, Tuple{Float32, Int64}}())
        @test Awale.MCTS.select_puct(mcts, root) == 2
    end

    @testset "repetition handling respects the configured policy" begin
        s = Awale.initial_state()
        repeated_draw = Awale.GameState(s.board, s.to_move, s.captured, s.history_hash, Awale.GameConfig(repetition=:draw_on_repeat), Set([s.history_hash]))
        repeated_score = Awale.GameState(s.board, s.to_move, (UInt8(25), UInt8(20)), s.history_hash, Awale.GameConfig(repetition=:score_diff), Set([s.history_hash]))
        repeated_revert = Awale.GameState(s.board, s.to_move, s.captured, s.history_hash, Awale.GameConfig(repetition=:revert), Set([s.history_hash]))

        @test Awale.is_terminal(repeated_draw)
        @test Awale.reward(repeated_draw) == 0.0f0
        @test Awale.Evaluation.result_from_terminal_state(repeated_draw) == 0

        @test Awale.is_terminal(repeated_score)
        @test Awale.reward(repeated_score) == 1.0f0
        @test Awale.Evaluation.result_from_terminal_state(repeated_score) == 1

        @test !Awale.is_terminal(repeated_revert)
    end

    @testset "training iteration populates replay buffer and performs updates" begin
        rng = MersenneTwister(11)
        model = Awale.create_model()
        optimizer = Flux.setup(Flux.Adam(1.0f-3), model)
        replay_buffer = Awale.ReplayBuffers.ReplayBuffer(256)
        mcts = Awale.MCTSSearch(model, 1.5f0, Dict{UInt64, Tuple{Float32, Int64}}())

        loss = Awale.run_training_iteration(
            mcts,
            optimizer,
            model,
            replay_buffer;
            n_games=1,
            sims=1,
            batch_size=8,
            updates_per_iteration=2,
            temperature_moves=2,
            rng=rng,
        )

        @test isfinite(loss)
        @test length(replay_buffer) > 0
    end

    @testset "training snapshot policy preserves milestone checkpoints" begin
        train_module = Module(:TrainSmoke)
        Core.eval(train_module, :(include(path) = Base.include($(train_module), path)))
        Base.include(train_module, joinpath(@__DIR__, "..", "train.jl"))

        @test train_module.should_save_snapshot(1, 25, [1, 5, 10, 25])
        @test train_module.should_save_snapshot(5, 25, [1, 5, 10, 25])
        @test train_module.should_save_snapshot(10, 25, [1, 5, 10, 25])
        @test train_module.should_save_snapshot(25, 25, [1, 5, 10, 25])
        @test !train_module.should_save_snapshot(2, 25, [1, 5, 10, 25])
    end

    @testset "entrypoint scripts load without executing main during tests" begin
        train_module = Module(:TrainSmoke2)
        eval_module = Module(:EvalSmoke)
        play_module = Module(:PlaySmoke)
        arena_module = Module(:ArenaSmoke)

        Core.eval(train_module, :(include(path) = Base.include($(train_module), path)))
        Core.eval(eval_module, :(include(path) = Base.include($(eval_module), path)))
        Core.eval(play_module, :(include(path) = Base.include($(play_module), path)))
        Core.eval(arena_module, :(include(path) = Base.include($(arena_module), path)))

        Base.include(train_module, joinpath(@__DIR__, "..", "train.jl"))
        Base.include(eval_module, joinpath(@__DIR__, "..", "baseline_eval.jl"))
        Base.include(play_module, joinpath(@__DIR__, "..", "play.jl"))
        Base.include(arena_module, joinpath(@__DIR__, "..", "checkpoint_arena.jl"))

        @test isdefined(train_module, :main)
        @test isdefined(eval_module, :main)
        @test isdefined(play_module, :main)
        @test isdefined(arena_module, :main)
        @test arena_module.checkpoint_label(5) == "iter_5"
    end
end
