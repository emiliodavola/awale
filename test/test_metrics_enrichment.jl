using Test
using TOML
using .Awale

@testset "Metrics enrichment" begin
    @testset "PromotionRecord TOML roundtrip" begin
        # Construct enriched record with all fields (including random_anchor_wr)
        record = Awale.Metrics.PromotionRecord(
            100,                                   # iteration
            58.5,                                  # win_rate_vs_best
            10,                                    # wins
            7,                                     # losses
            3,                                     # draws
            62.0,                                  # random_anchor_wr
            58.5,                                  # promotion_score
            100,                                   # gap_since_last (first promotion)
            1,                                     # total_promotions_at_event
            1550.0,                                # elo_candidate
            1500.0,                                # elo_best
            "2026-07-29T13:00:00Z",                # timestamp
        )

        # Serialize to TOML string (matching save_promotion_history format)
        toml_lines = String[]
        push!(toml_lines, "iteration = $(record.iteration)")
        push!(toml_lines, "win_rate_vs_best = $(record.win_rate_vs_best)")
        push!(toml_lines, "wins = $(record.wins)")
        push!(toml_lines, "losses = $(record.losses)")
        push!(toml_lines, "draws = $(record.draws)")
        push!(toml_lines, "random_anchor_wr = $(record.random_anchor_wr)")
        push!(toml_lines, "promotion_score = $(record.promotion_score)")
        push!(toml_lines, "gap_since_last = $(record.gap_since_last)")
        push!(toml_lines, "total_promotions_at_event = $(record.total_promotions_at_event)")
        push!(toml_lines, "elo_candidate = $(record.elo_candidate)")
        push!(toml_lines, "elo_best = $(record.elo_best)")
        push!(toml_lines, "timestamp = \"$(record.timestamp)\"")
        toml_str = join(toml_lines, "\n")

        # Parse back
        parsed = TOML.parse(toml_str)

        # Reconstruct
        restored = Awale.Metrics.PromotionRecord(
            Int(parsed["iteration"]),
            Float64(parsed["win_rate_vs_best"]),
            Int(parsed["wins"]),
            Int(parsed["losses"]),
            Int(parsed["draws"]),
            Float64(parsed["random_anchor_wr"]),
            Float64(parsed["promotion_score"]),
            Int(parsed["gap_since_last"]),
            Int(parsed["total_promotions_at_event"]),
            Float64(parsed["elo_candidate"]),
            Float64(parsed["elo_best"]),
            String(parsed["timestamp"]),
        )

        @test restored.iteration == record.iteration
        @test restored.win_rate_vs_best == record.win_rate_vs_best
        @test restored.wins == record.wins
        @test restored.losses == record.losses
        @test restored.draws == record.draws
        @test restored.random_anchor_wr == record.random_anchor_wr
        @test restored.promotion_score == record.promotion_score
        @test restored.gap_since_last == record.gap_since_last
        @test restored.total_promotions_at_event == record.total_promotions_at_event
        @test restored.elo_candidate == record.elo_candidate
        @test restored.elo_best == record.elo_best
        @test restored.timestamp == record.timestamp
    end

    @testset "PromotionRecord roundtrip without optional field" begin
        # Roundtrip with random_anchor_wr = nothing (field absent in TOML)
        record = Awale.Metrics.PromotionRecord(
            50, 55.0, 8, 6, 2, nothing, 55.0, 50, 2, 1520.0, 1500.0, "2026-07-28T10:00:00Z",
        )

        toml_lines = String[]
        push!(toml_lines, "iteration = $(record.iteration)")
        push!(toml_lines, "win_rate_vs_best = $(record.win_rate_vs_best)")
        push!(toml_lines, "wins = $(record.wins)")
        push!(toml_lines, "losses = $(record.losses)")
        push!(toml_lines, "draws = $(record.draws)")
        # random_anchor_wr intentionally omitted (matches save_promotion_history behavior)
        push!(toml_lines, "promotion_score = $(record.promotion_score)")
        push!(toml_lines, "gap_since_last = $(record.gap_since_last)")
        push!(toml_lines, "total_promotions_at_event = $(record.total_promotions_at_event)")
        push!(toml_lines, "elo_candidate = $(record.elo_candidate)")
        push!(toml_lines, "elo_best = $(record.elo_best)")
        push!(toml_lines, "timestamp = \"$(record.timestamp)\"")
        toml_str = join(toml_lines, "\n")

        parsed = TOML.parse(toml_str)

        restored = Awale.Metrics.PromotionRecord(
            Int(parsed["iteration"]),
            Float64(parsed["win_rate_vs_best"]),
            Int(parsed["wins"]),
            Int(parsed["losses"]),
            Int(parsed["draws"]),
            get(parsed, "random_anchor_wr", nothing),
            Float64(parsed["promotion_score"]),
            Int(parsed["gap_since_last"]),
            Int(parsed["total_promotions_at_event"]),
            Float64(parsed["elo_candidate"]),
            Float64(parsed["elo_best"]),
            String(parsed["timestamp"]),
        )

        @test restored.random_anchor_wr === nothing
        @test restored.iteration == 50
        @test restored.gap_since_last == 50
        @test restored.total_promotions_at_event == 2
        @test restored.elo_candidate == 1520.0
        @test restored.elo_best == 1500.0
        @test restored.timestamp == "2026-07-28T10:00:00Z"
    end

    @testset "Elo K=64 vs K=32 produces larger delta" begin
        elo_64 = Awale.Metrics.EloTracker(candidate=1500.0, best=1500.0, k=64.0)
        elo_32 = Awale.Metrics.EloTracker(candidate=1500.0, best=1500.0, k=32.0)

        # 55% win rate over 20 games (11 wins, 9 losses, 0 draws)
        Awale.Metrics.update_elo!(elo_64, 11, 9, 0, 1)
        Awale.Metrics.update_elo!(elo_32, 11, 9, 0, 1)

        delta_64 = abs(elo_64.candidate_rating - 1500.0)
        delta_32 = abs(elo_32.candidate_rating - 1500.0)

        @test delta_64 > delta_32
        # delta = K * (score - expected) = K * (0.55 - 0.50) = K * 0.05
        # K=64 → delta = 3.2; K=32 → delta = 1.6
        @test delta_64 ≈ 3.2 atol=0.02
        @test delta_32 ≈ 1.6 atol=0.02
    end

    @testset "Value calibration — perfect predictions" begin
        v_pred = Float32[0.5, -0.3, 0.8, 0.1, -0.9]
        v_target = Float32[0.5, -0.3, 0.8, 0.1, -0.9]

        result = Awale.Metrics.compute_value_calibration(v_pred, v_target)

        @test result.mae ≈ 0.0 atol=1e-6
        @test result.pearson_r ≈ 1.0 atol=1e-6
        @test result.spearman_rho ≈ 1.0 atol=1e-6
    end

    @testset "Value calibration — inverted predictions" begin
        v_pred = Float32[0.5, -0.3, 0.8, 0.1, -0.9]
        v_target = Float32[-0.5, 0.3, -0.8, -0.1, 0.9]

        result = Awale.Metrics.compute_value_calibration(v_pred, v_target)

        @test result.pearson_r ≈ -1.0 atol=1e-6
        @test result.spearman_rho ≈ -1.0 atol=1e-6
    end

    @testset "Value calibration — degenerate input" begin
        v_pred = Float32[0.5, 0.5, 0.5]
        v_target = Float32[0.1, 0.2, 0.3]

        result = Awale.Metrics.compute_value_calibration(v_pred, v_target)

        @test isnan(result.mae)
        @test isnan(result.pearson_r)
        @test isnan(result.spearman_rho)

        # Also test degenerate target
        v_pred2 = Float32[0.1, 0.2, 0.3]
        v_target2 = Float32[0.5, 0.5, 0.5]

        result2 = Awale.Metrics.compute_value_calibration(v_pred2, v_target2)

        @test isnan(result2.mae)
        @test isnan(result2.pearson_r)
        @test isnan(result2.spearman_rho)
    end

    @testset "Historical summary formatting" begin
        # Build a mock ProgressTracker with promotion records
        pt = Awale.Metrics.ProgressTracker()
        pt.last_best_iter = 30
        pt.total_promotions = 3
        pt.inter_promotion_gaps = [30, 25]  # first gap=30, second gap=25
        pt.longest_streak = 15

        # Push records manually (bypass record_promotion! to avoid auto-gap logic)
        r1 = Awale.Metrics.PromotionRecord(30, 52.1, 10, 8, 2, 60.0, 52.1, 30, 1, 1520.0, 1500.0, "2026-07-28T10:00:00Z")
        r2 = Awale.Metrics.PromotionRecord(55, 62.0, 12, 6, 2, 65.0, 62.0, 25, 2, 1560.0, 1520.0, "2026-07-29T08:00:00Z")
        r3 = Awale.Metrics.PromotionRecord(85, 58.1, 11, 7, 2, 63.0, 58.1, 30, 3, 1580.0, 1560.0, "2026-07-29T14:00:00Z")
        pt.promotions = [r1, r2, r3]

        # Capture stdout
        log = Pipe()
        redirect_stdout(log) do
            Awale.Metrics.print_historical_summary(pt)
        end
        close(log.in)
        output = read(log, String)

        @test occursin("Promotion History Summary", output)
        @test occursin("Total promotions          : 3", output)
        @test occursin("Avg win rate              : 57.4%", output)  # (52.1+62.0+58.1)/3 = 57.4
        @test occursin("Median win rate           : 58.1%", output)
        @test occursin("Best promotion            : 62.0% (iter 55)", output)
        @test occursin("Closest promotion         : 52.1% (iter 30)", output)
        @test occursin("Avg gap between promos    : 27.5 iters", output)  # (30+25)/2 = 27.5
        @test occursin("Median gap                : 27.5 iters", output)
        @test occursin("Longest streak no promo   : 15 iters", output)
    end

    @testset "Historical summary — no promotions" begin
        pt = Awale.Metrics.ProgressTracker()
        log = Pipe()
        redirect_stdout(log) do
            Awale.Metrics.print_historical_summary(pt)
        end
        close(log.in)
        output = read(log, String)

        @test occursin("Total promotions          : 0", output)
        @test occursin("Longest streak no promo   : 0 iters", output)
        # Should not contain avg/median/best/closest when no records
        @test !occursin("Avg win rate", output)
        @test !occursin("Avg gap", output)
    end

    @testset "EloTracker default k is 64" begin
        et = Awale.Metrics.EloTracker()
        @test et.k == 64.0
    end
end
