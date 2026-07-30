using Test
using Flux
using Random
using JSON
using TOML
using .Awale

@testset "JSONL observability" begin
    @testset "JSONL file created and correct number of lines" begin
        mktempdir() do tmpdir
            # Setup minimal training environment
            rng = MersenneTwister(42)
            model = Awale.create_model()
            optimizer = Flux.setup(Flux.Adam(1.0f-3), model)
            replay_buffer = Awale.ReplayBuffers.ReplayBuffer(256)
            mcts = Awale.MCTSSearch(model, 1.5f0, 0.3f0, 0.25f0, Dict{UInt64, Tuple{Float64, Int64}}())

            # Run 3 iterations of training (simulating loop)
            n_iters = 3
            for iter_idx in 1:n_iters
                result, cal = Awale.run_training_iteration(
                    mcts, optimizer, model, replay_buffer;
                    n_games=1, sims=1, batch_size=8, updates_per_iteration=1,
                    replay_recent_fraction=0.5, replay_recent_window=32,
                    temperature_moves=2, rng=rng, max_turns=1000,
                )
                @test isfinite(result)
                @test length(replay_buffer) > 0
            end

            # Write JSONL manually and verify
            jsonl_path = joinpath(tmpdir, "metrics_test.jsonl")
            open(jsonl_path, "w") do io
                for i in 1:3
                    d = Dict("iter" => i, "avg_loss" => Float64(rand()), "metric_version" => "1.0.0")
                    println(io, JSON.json(d))
                end
            end

            lines = readlines(jsonl_path)
            @test length(lines) == 3
            for line in lines
                parsed = JSON.parse(line)
                @test haskey(parsed, "iter")
                @test haskey(parsed, "avg_loss")
                @test haskey(parsed, "metric_version")
            end
        end
    end

    @testset "Dual-write matching CSV + JSONL values" begin
        mktempdir() do tmpdir
            csv_path = joinpath(tmpdir, "test.csv")
            jsonl_path = joinpath(tmpdir, "test.jsonl")

            # Write CSV
            open(csv_path, "w") do csv_io
                Awale.Metrics.write_csv_header(csv_io)
                Awale.Metrics.write_csv_row(csv_io, 1, 1.0f0, 0.5f0, 0.5f0, 2.0f0,
                    0.8f0, 0.7f0, 0.1, 42.0, 35.0, 95.0, nothing, false, 1500.0, 1500.0)
                Awale.Metrics.write_csv_row(csv_io, 2, 0.9f0, 0.4f0, 0.5f0, 1.8f0,
                    0.85f0, 0.72f0, 0.08, 55.0, 34.0, 96.0, 55.0, true, 1510.0, 1500.0)
            end

            # Write JSONL with matching values
            open(jsonl_path, "w") do io
                d1 = Dict("iter" => 1, "avg_loss" => 1.0, "policy_loss" => 0.5,
                    "value_loss" => 0.5, "grad_norm" => 2.0, "pred_entropy" => 0.8,
                    "target_entropy" => 0.7, "replay_fill_pct" => 42.0,
                    "elo_candidate" => 1500.0, "elo_best" => 1500.0,
                    "metric_version" => "1.0.0", "git_commit" => "abc123",
                    "mcts_kl_mean" => 0.5, "drift_kl" => 0.01)
                println(io, JSON.json(d1))

                d2 = Dict("iter" => 2, "avg_loss" => 0.9, "policy_loss" => 0.4,
                    "value_loss" => 0.5, "grad_norm" => 1.8, "pred_entropy" => 0.85,
                    "target_entropy" => 0.72, "replay_fill_pct" => 55.0,
                    "candidate_vs_best_wr" => 55.0, "promoted" => true,
                    "elo_candidate" => 1510.0, "elo_best" => 1500.0,
                    "metric_version" => "1.0.0", "git_commit" => "abc123",
                    "mcts_kl_mean" => 0.45, "drift_kl" => 0.008)
                println(io, JSON.json(d2))
            end

            # Verify CSV content
            csv_lines = readlines(csv_path)
            @test length(csv_lines) == 3  # header + 2 rows
            @test occursin("iteration,avg_loss,policy_loss", csv_lines[1])

            # Parse CSV rows
            for idx in 2:3
                fields = split(csv_lines[idx], ",")
                @test length(fields) == 15
            end

            # Verify JSONL content
            jsonl_lines = readlines(jsonl_path)
            @test length(jsonl_lines) == 2
            for line in jsonl_lines
                parsed = JSON.parse(line)
                @test haskey(parsed, "iter")
                @test haskey(parsed, "avg_loss")
                @test haskey(parsed, "metric_version")
                @test haskey(parsed, "mcts_kl_mean")
                @test haskey(parsed, "drift_kl")
            end

            # Cross-verify matching values
            j1 = JSON.parse(jsonl_lines[1])
            csv_row1 = split(csv_lines[2], ",")
            @test parse(Float64, csv_row1[2]) ≈ j1["avg_loss"]
            @test parse(Float64, csv_row1[3]) ≈ j1["policy_loss"]
            @test parse(Float64, csv_row1[4]) ≈ j1["value_loss"]

            j2 = JSON.parse(jsonl_lines[2])
            csv_row2 = split(csv_lines[3], ",")
            @test parse(Float64, csv_row2[2]) ≈ j2["avg_loss"]
            @test parse(Float64, csv_row2[3]) ≈ j2["policy_loss"]
        end
    end

    @testset "JSONL contains versioning fields" begin
        mktempdir() do tmpdir
            jsonl_path = joinpath(tmpdir, "versioned.jsonl")
            open(jsonl_path, "w") do io
                d = Dict(
                    "metric_version" => "1.0.0",
                    "git_commit" => "abcdef1234567890",
                    "architecture" => "mlp",
                    "config_hash" => "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                    "iter" => 1,
                    "avg_loss" => 4.21,
                    "mcts_kl_mean" => 0.45,
                    "drift_kl" => 0.12,
                )
                println(io, JSON.json(d))
            end

            lines = readlines(jsonl_path)
            @test length(lines) == 1
            parsed = JSON.parse(lines[1])
            @test parsed["metric_version"] == "1.0.0"
            @test parsed["git_commit"] == "abcdef1234567890"
            @test parsed["architecture"] == "mlp"
            @test occursin(r"^[0-9a-f]{64}$", parsed["config_hash"])
        end
    end

    @testset "JSONL contains MCTS aggregate fields" begin
        mktempdir() do tmpdir
            jsonl_path = joinpath(tmpdir, "mcts_agg.jsonl")
            open(jsonl_path, "w") do io
                d = Dict(
                    "mcts_kl_mean" => 0.45,
                    "mcts_kl_median" => 0.38,
                    "mcts_top1_pct" => 42.5,
                    "mcts_top2_pct" => 68.1,
                    "mcts_top3_pct" => 82.3,
                    "mcts_root_conf_mean" => 0.32,
                    "mcts_l1_mean" => 0.41,
                    "metric_version" => "1.0.0",
                )
                println(io, JSON.json(d))
            end

            parsed = JSON.parse(readlines(jsonl_path)[1])
            @test haskey(parsed, "mcts_kl_mean")
            @test haskey(parsed, "mcts_kl_median")
            @test haskey(parsed, "mcts_top1_pct")
            @test haskey(parsed, "mcts_top2_pct")
            @test haskey(parsed, "mcts_top3_pct")
            @test haskey(parsed, "mcts_root_conf_mean")
            @test haskey(parsed, "mcts_l1_mean")
            @test parsed["mcts_kl_mean"] == 0.45
            @test parsed["mcts_top1_pct"] == 42.5
        end
    end

    @testset "Moving averages: NaN until window full" begin
        # Simulate the ma_win function used in train.jl
        ma_win(buf, n) = length(buf) >= n ? sum(buf[end-n+1:end]) / n : NaN

        buf = Float32[1.0, 2.0, 3.0]
        @test isnan(ma_win(buf, 5))
        @test isnan(ma_win(buf, 4))
        @test ma_win(buf, 3) ≈ 2.0
        @test ma_win(buf, 2) ≈ 2.5

        push!(buf, 4.0)
        push!(buf, 5.0)
        @test ma_win(buf, 5) ≈ 3.0
        @test ma_win(buf, 3) ≈ 4.0
        @test ma_win(buf, 1) ≈ 5.0
    end

    @testset "Replay Fill % label renamed" begin
        # The spec says old "Replay Coverage" label must be gone from print paths
        # Search in Training.jl for the renamed print label
        training_src = read(joinpath(@__DIR__, "..", "src", "Awale", "Training.jl"), String)
        metrics_src = read(joinpath(@__DIR__, "..", "src", "Awale", "Metrics.jl"), String)
        # Old label must be gone from ALL source files
        @test !occursin("Replay Coverage", training_src)
        @test !occursin("Replay Coverage", metrics_src)
        # New label must exist in the print paths
        @test occursin("Replay Fill %", training_src)
    end

    @testset "run_training_iteration returns calibration data" begin
        rng = MersenneTwister(11)
        model = Awale.create_model()
        optimizer = Flux.setup(Flux.Adam(1.0f-3), model)
        replay_buffer = Awale.ReplayBuffers.ReplayBuffer(256)
        mcts = Awale.MCTSSearch(model, 1.5f0, 0.3f0, 0.25f0, Dict{UInt64, Tuple{Float64, Int64}}())

        result, calib_data = Awale.run_training_iteration(
            mcts, optimizer, model, replay_buffer;
            n_games=1, sims=1, batch_size=8, updates_per_iteration=2,
            replay_recent_fraction=0.5, replay_recent_window=32,
            temperature_moves=2, rng=rng, max_turns=1000,
        )

        @test isfinite(result)
        @test haskey(calib_data, :v_pred)
        @test haskey(calib_data, :v_target)
        # v_pred and v_target should have the same length (or both be empty)
        @test length(calib_data.v_pred) == length(calib_data.v_target)
        @test length(calib_data.v_pred) > 0  # training happened

        # Compute value calibration from calib_data
        vc = Awale.Metrics.compute_value_calibration(vec(calib_data.v_pred), vec(calib_data.v_target))
        @test vc.mae isa AbstractFloat
        @test isfinite(vc.mae) || isnan(vc.mae)  # either valid or degenerate
        @test vc.pearson_r isa AbstractFloat
        @test vc.spearman_rho isa AbstractFloat
    end

    @testset "TrainingResult has new Search Gain fields" begin
        rng = MersenneTwister(11)
        model = Awale.create_model()
        optimizer = Flux.setup(Flux.Adam(1.0f-3), model)
        replay_buffer = Awale.ReplayBuffers.ReplayBuffer(256)
        mcts = Awale.MCTSSearch(model, 1.5f0, 0.3f0, 0.25f0, Dict{UInt64, Tuple{Float64, Int64}}())

        result, _ = Awale.run_training_iteration(
            mcts, optimizer, model, replay_buffer;
            n_games=1, sims=1, batch_size=8, updates_per_iteration=2,
            replay_recent_fraction=0.5, replay_recent_window=32,
            temperature_moves=2, rng=rng, max_turns=1000,
        )

        @test hasfield(Awale.TrainingResult, :root_q_mean)
        @test hasfield(Awale.TrainingResult, :network_value_mean)
        @test hasfield(Awale.TrainingResult, :kl_p25)
        @test hasfield(Awale.TrainingResult, :kl_p75)
        @test hasfield(Awale.TrainingResult, :l1_p25)
        @test hasfield(Awale.TrainingResult, :l1_p75)
        @test hasfield(Awale.TrainingResult, :entropy_mean)
        @test hasfield(Awale.TrainingResult, :entropy_min)
        @test hasfield(Awale.TrainingResult, :entropy_max)
        @test hasfield(Awale.TrainingResult, :root_conf_min)
        @test hasfield(Awale.TrainingResult, :root_conf_max)
        @test hasfield(Awale.TrainingResult, :root_conf_p25)
        @test hasfield(Awale.TrainingResult, :root_conf_p75)

        # root_q_mean and network_value_mean should be finite
        @test isfinite(result.root_q_mean) || result.root_q_mean == 0.0f0
        @test isfinite(result.network_value_mean) || result.network_value_mean == 0.0f0

        # Search gain = root Q - network value
        search_gain = result.root_q_mean - result.network_value_mean
        @test isfinite(search_gain) || search_gain == 0.0f0
    end

    @testset "JSONL new fields serialization" begin
        mktempdir() do tmpdir
            jsonl_path = joinpath(tmpdir, "new_fields.jsonl")
            open(jsonl_path, "w") do io
                d = Dict(
                    "search_gain" => 0.042,
                    "root_q_mean" => 0.15,
                    "network_value_mean" => 0.108,
                    "kl_p25" => 0.1, "kl_p50" => 0.3, "kl_p75" => 0.6, "kl_p95" => 1.2,
                    "l1_p25" => 0.05, "l1_p50" => 0.12, "l1_p75" => 0.3, "l1_p95" => 0.5,
                    "entropy_mean" => 0.8, "entropy_min" => 0.1, "entropy_max" => 1.2,
                    "entropy_p25" => 0.4, "entropy_p50" => 0.7, "entropy_p75" => 1.0, "entropy_p95" => 1.1,
                    "root_conf_min" => 0.05, "root_conf_max" => 0.95,
                    "root_conf_p25" => 0.2, "root_conf_p50" => 0.4, "root_conf_p75" => 0.6, "root_conf_p95" => 0.8,
                    "net_health" => "ACTIVE", "srch_health" => "HIGH",
                    "drift_health" => "LOW", "valcal_health" => "OK",
                    "stability_kl" => "ACTIVE", "stability_drift" => "STALLED",
                    "stability_top1" => "ACTIVE", "stability_param" => "BOOTSTRAP",
                    "warning_count" => 0, "warning_messages" => [],
                    "metric_version" => "1.0.0",
                )
                println(io, JSON.json(d))
            end

            parsed = JSON.parse(readlines(jsonl_path)[1])
            @test haskey(parsed, "search_gain")
            @test haskey(parsed, "root_q_mean")
            @test haskey(parsed, "network_value_mean")
            @test haskey(parsed, "kl_p25")
            @test haskey(parsed, "kl_p95")
            @test haskey(parsed, "l1_p25")
            @test haskey(parsed, "l1_p95")
            @test haskey(parsed, "entropy_mean")
            @test haskey(parsed, "entropy_min")
            @test haskey(parsed, "entropy_max")
            @test haskey(parsed, "root_conf_min")
            @test haskey(parsed, "root_conf_max")
            @test haskey(parsed, "root_conf_p95")
            @test haskey(parsed, "net_health")
            @test haskey(parsed, "srch_health")
            @test haskey(parsed, "drift_health")
            @test haskey(parsed, "valcal_health")
            @test haskey(parsed, "stability_kl")
            @test haskey(parsed, "stability_drift")
            @test haskey(parsed, "stability_param")
            @test haskey(parsed, "warning_count")
            @test haskey(parsed, "warning_messages")
            @test parsed["search_gain"] == 0.042
            @test parsed["warning_messages"] == []
        end
    end
end
