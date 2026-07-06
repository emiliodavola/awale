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

    @testset "training snapshot policy writes only milestone snapshots" begin
        train_module = Module(:TrainSmoke)
        Core.eval(Main, :(TrainSmoke = $train_module))
        Core.eval(train_module, :(include(path) = Base.include($(train_module), path)))
        Base.include(train_module, joinpath(@__DIR__, "..", "train.jl"))

        @test train_module.should_save_snapshot(1, 25, 25)
        @test train_module.should_save_snapshot(2, 25, 25)
        @test train_module.should_save_snapshot(4, 25, 25)
        @test train_module.should_save_snapshot(8, 25, 25)
        @test train_module.should_save_snapshot(16, 25, 25)
        @test train_module.should_save_snapshot(25, 25, 25)
        @test !train_module.should_save_snapshot(3, 25, 25)
        @test !train_module.should_save_snapshot(24, 25, 25)
        @test !train_module.should_save_snapshot(27, 27, 25)
        @test train_module.decided_win_rate((wins=6, losses=4, draws=0, avg_turns=0.0)) == 60.0
        @test train_module.decided_win_rate((wins=0, losses=0, draws=10, avg_turns=0.0)) == 50.0
        @test train_module.selection_gate_status(56.0, NamedTuple[]).promoted
        @test !train_module.selection_gate_status(54.0, NamedTuple[]).promoted
        @test !train_module.selection_gate_status(nothing, [(name="random", results=(wins=0, losses=0, draws=0, avg_turns=0.0), decided_win_rate=49.0)]).promoted
        @test_throws ArgumentError train_module.validate_selection_config(0, [0, 2], 1)
        @test_throws ArgumentError train_module.validate_selection_config(2, Int[], 1)
        @test_throws ArgumentError train_module.validate_selection_config(2, [0, 2], 0)

        mktemp() do path, io
            write(io, "last_iter = 7\nbest_win_rate = 61.5\n")
            flush(io)
            close(io)
            last_iter, best_selection_score = train_module.read_training_state(path)
            @test last_iter == 7
            @test best_selection_score == 61.5
        end

        mktempdir() do tmpdir
            best_path = joinpath(tmpdir, "model_best.bin")
            train_module.BEST_CHECKPOINT_PATH = best_path
            initial_model = train_module.Awale.create_model()
            candidate_model = train_module.Awale.create_model()
            train_module.save_model(initial_model, best_path)
            before_bytes = read(best_path)
            score_ref = Ref(12.0)

            blocked = (promoted=false, promotion_score=77.0)
            @test !train_module.maybe_promote_best!(candidate_model, score_ref, blocked)
            @test score_ref[] == 12.0
            @test read(best_path) == before_bytes

            promoted = (promoted=true, promotion_score=77.0)
            @test train_module.maybe_promote_best!(candidate_model, score_ref, promoted)
            @test score_ref[] == 77.0
            @test read(best_path) != before_bytes

            train_module.BEST_TARGET_SIMS = 0
            train_module.BEST_PROMOTION_GAMES = 2
            train_module.BEST_OPENING_PLIES = [0]
            train_module.BEST_OPENINGS_PER_PLY = 1
            train_module.BEST_SELECTION_SEED = 123
            train_module.USE_RANDOM_ANCHOR = true
            train_module.USE_HEURISTIC_ANCHOR = false
            train_module.save_model(initial_model, best_path)
            promotion_a = train_module.evaluate_best_promotion(candidate_model)
            promotion_b = train_module.evaluate_best_promotion(candidate_model)
            @test promotion_a.promoted == promotion_b.promoted
            @test promotion_a.promotion_score == promotion_b.promotion_score
            @test promotion_a.current_best_rate == promotion_b.current_best_rate
            @test promotion_a.anchor_reports == promotion_b.anchor_reports
        end

        mktempdir() do tmpdir
            invalid_checkpoint_dir = joinpath(tmpdir, "invalid-checkpoints")
            train_module.CHECKPOINT_DIR = invalid_checkpoint_dir
            train_module.BEST_PROMOTION_GAMES = 0
            @test_throws ArgumentError train_module.main(String["--reset"])
            @test !isdir(invalid_checkpoint_dir)
        end

        mktempdir() do tmpdir
            checkpoint_dir = joinpath(tmpdir, "checkpoints")
            mkpath(checkpoint_dir)
            train_module.CHECKPOINT_DIR = checkpoint_dir
            train_module.LAST_CHECKPOINT_PATH = joinpath(checkpoint_dir, "model_last.bin")
            train_module.BEST_CHECKPOINT_PATH = joinpath(checkpoint_dir, "model_best.bin")
            train_module.CHECKPOINT_PATH = joinpath(checkpoint_dir, "model_final.bin")
            train_module.STATE_PATH = joinpath(checkpoint_dir, "training_state.toml")
            train_module.NUM_ITERATIONS = 27
            train_module.GAMES_PER_ITERATION = 1
            train_module.SIMS_PER_MOVE = 1
            train_module.BATCH_SIZE = 8
            train_module.UPDATES_PER_ITERATION = 1
            train_module.TEMPERATURE_MOVES = 2
            train_module.CHECKPOINT_EVERY = 25
            train_module.EVAL_GAMES = 2
            train_module.SIMS_PER_EVAL = 1
            train_module.BEST_TARGET_SIMS = 1
            train_module.BEST_PROMOTION_GAMES = 2
            train_module.BEST_OPENING_PLIES = [0]
            train_module.BEST_OPENINGS_PER_PLY = 1
            train_module.USE_RANDOM_ANCHOR = false
            train_module.USE_HEURISTIC_ANCHOR = false

            Random.seed!(1234)
            train_module.save_model(train_module.Awale.create_model(), train_module.LAST_CHECKPOINT_PATH)
            train_module.write_training_state(train_module.STATE_PATH, 24, 0.0)
            first_output = mktemp() do path, io
                redirect_stdout(io) do
                    train_module.main(String[])
                end
                flush(io)
                close(io)
                read(path, String)
            end

            @test occursin("Reanudando desde la iteración 24", first_output)
            @test occursin("Best-selection target: 1 sims, 2 games, 1 openings, threshold", first_output)
            @test isfile(joinpath(checkpoint_dir, "model_iter_25.bin"))
            @test !isfile(joinpath(checkpoint_dir, "model_iter_26.bin"))
            @test !isfile(joinpath(checkpoint_dir, "model_iter_27.bin"))
            @test isfile(joinpath(checkpoint_dir, "model_final.bin"))

            second_output = mktemp() do path, io
                redirect_stdout(io) do
                    train_module.main(String[])
                end
                flush(io)
                close(io)
                read(path, String)
            end
            @test occursin("--- Entrenamiento ya completado. ---", second_output)
        end
    end

    @testset "entrypoint scripts load without executing main during tests" begin
        train_module = Module(:TrainSmoke2)
        eval_module = Module(:EvalSmoke)
        play_module = Module(:PlaySmoke)
        arena_module = Module(:ArenaSmoke)

        Core.eval(Main, :(TrainSmoke2 = $train_module))
        Core.eval(Main, :(EvalSmoke = $eval_module))
        Core.eval(Main, :(PlaySmoke = $play_module))
        Core.eval(Main, :(ArenaSmoke = $arena_module))
        Core.eval(train_module, :(include(path) = Base.include($(train_module), path)))
        Core.eval(eval_module, :(include(path) = Base.include($(eval_module), path)))
        Core.eval(play_module, :(include(path) = Base.include($(play_module), path)))
        Core.eval(arena_module, :(include(path) = Base.include($(arena_module), path)))

        mktempdir() do tmpdir
            cd(tmpdir) do
                Base.include(train_module, joinpath(@__DIR__, "..", "train.jl"))
                Base.include(eval_module, joinpath(@__DIR__, "..", "baseline_eval.jl"))
                Base.include(play_module, joinpath(@__DIR__, "..", "play.jl"))
                Base.include(arena_module, joinpath(@__DIR__, "..", "checkpoint_arena.jl"))
                model = Awale.create_model()
                @test length(Awale.predict(model, Awale.initial_state())[1]) == 6
            end
        end

        @test isdefined(train_module, :main)
        @test isdefined(eval_module, :main)
        @test isdefined(play_module, :main)
        @test isdefined(arena_module, :main)
        @test arena_module.checkpoint_label(5) == "iter_5"

        openings_a = arena_module.generate_opening_suite(plies=[0, 2, 4, 6], seed=123, openings_per_ply=2)
        openings_b = arena_module.generate_opening_suite(plies=[0, 2, 4, 6], seed=123, openings_per_ply=2)
        @test length(openings_a) == 8
        @test arena_module.winner_label(5, 10, (wins=3, losses=2, draws=1)) == "iter_5"
        @test arena_module.winner_label(5, 10, (wins=2, losses=3, draws=1)) == "iter_10"
        @test arena_module.winner_label(5, 10, (wins=2, losses=2, draws=4)) == "tie"
        @test occursin("Who wins", arena_module.format_header())
        @test occursin("iter_5", arena_module.format_duel_result(5, 10, (wins=3, losses=2, draws=1, avg_turns=42.5)))
        @test [arena_module.Awale.serialize_state(s) for s in openings_a] == [arena_module.Awale.serialize_state(s) for s in openings_b]

        original_checkpoint_dir = arena_module.CHECKPOINT_DIR
        mktempdir() do tmpdir
            for iter in (5, 10, 25, 26, 27)
                touch(joinpath(tmpdir, "model_iter_$(iter).bin"))
            end
            touch(joinpath(tmpdir, "model_last.bin"))
            touch(joinpath(tmpdir, "model_best.bin"))
            touch(joinpath(tmpdir, "model_final.bin"))
            arena_module.CHECKPOINT_DIR = tmpdir
            @test arena_module.available_matchups() == [(5, 10), (10, 25), (25, 26), (26, 27)]
        end

        mktempdir() do tmpdir
            arena_module.CHECKPOINT_DIR = tmpdir
            arena_module.Awale.Model.save_model(arena_module.Awale.create_model(), joinpath(tmpdir, "model_iter_5.bin"))
            arena_module.Awale.Model.save_model(arena_module.Awale.create_model(), joinpath(tmpdir, "model_iter_10.bin"))
            openings = arena_module.generate_opening_suite(plies=[0], seed=321, openings_per_ply=1)
            duel_a = arena_module.run_duel(5, 10; sims=0, games=2, openings=openings)
            duel_b = arena_module.run_duel(5, 10; sims=0, games=2, openings=openings)
            @test duel_a == duel_b
        end

        mktempdir() do tmpdir
            arena_module.CHECKPOINT_DIR = joinpath(tmpdir, "missing")
            @test arena_module.existing_checkpoint_labels() == Any[]
            @test arena_module.available_matchups() == Tuple{Int, Int}[]
            output = mktemp() do path, io
                redirect_stdout(io) do
                    arena_module.main()
                end
                flush(io)
                close(io)
                read(path, String)
            end
            @test occursin("No hay suficientes checkpoints compatibles para correr el arena.", output)
        end
        arena_module.CHECKPOINT_DIR = original_checkpoint_dir
    end
end
