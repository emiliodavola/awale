"""
    Metrics

Training-progress tracking: Elo ratings, promotion history, learning curve CSV,
and inter-iteration diagnostics logging.
"""
module Metrics

using ..State: GameState
using ..Model: encode_state
using Printf
using Statistics

export EloTracker, ProgressTracker, PromotionRecord, update_elo!, promote_elo!,
       write_csv_header, write_csv_row, print_progress_diagnostics, print_promotion_event,
       compute_value_calibration, print_historical_summary

# ── Elo ──────────────────────────────────────
"""
    EloTracker

Track Elo ratings for candidate vs best model comparison.

# Fields
- `candidate_rating::Float64` — Current Elo of the candidate model (updates each iteration)
- `best_rating::Float64` — Elo of the best (promoted) model (only updates on promotion)
- `k::Float64` — K-factor controlling rating change sensitivity (default: 64.0)
- `history::Vector{Tuple{Int, Float64, Float64}}` — (iteration, candidate_rating, best_rating) snapshots
"""
mutable struct EloTracker
    candidate_rating::Float64
    best_rating::Float64
    k::Float64
    history::Vector{Tuple{Int, Float64, Float64}}
end

"""
    EloTracker(; candidate=1500.0, best=1500.0, k=64.0) -> EloTracker

Construct an EloTracker with optional initial ratings and K-factor.
"""
function EloTracker(; candidate=1500.0, best=1500.0, k=64.0)
    return EloTracker(candidate, best, k, Tuple{Int, Float64, Float64}[])
end

"""
    update_elo!(tracker::EloTracker, wins::Int, losses::Int, draws::Int, iteration::Int)

Update the candidate Elo rating based on match results against the current best model.

Uses standard Elo: expected = 1/(1+10^((best−candidate)/400)), score = (wins+0.5·draws)/total,
Δ = K·(score−expected). Appends a snapshot to `tracker.history`.
"""
function update_elo!(tracker::EloTracker, wins::Int, losses::Int, draws::Int, iteration::Int)
    total = wins + losses + draws
    total == 0 && return
    score = (wins + 0.5 * draws) / total
    expected = 1.0 / (1.0 + 10.0 ^ ((tracker.best_rating - tracker.candidate_rating) / 400.0))
    delta = tracker.k * (score - expected)
    tracker.candidate_rating += delta
    push!(tracker.history, (iteration, tracker.candidate_rating, tracker.best_rating))
end

"""
    promote_elo!(tracker::EloTracker)

Promote the candidate rating to become the new best rating.
Called when a candidate model passes the promotion gate.
"""
function promote_elo!(tracker::EloTracker)
    tracker.best_rating = tracker.candidate_rating
end

# ── Promotion Record ─────────────────────────
"""
    PromotionRecord

Data for a single promotion event. Persisted to `promotion_history.toml` for resume support.

# Fields
- `iteration::Int` — Training iteration when the promotion occurred
- `win_rate_vs_best::Float64` — Candidate's decided win rate (%) against the previous best model
- `wins::Int`, `losses::Int`, `draws::Int` — Raw match results
- `random_anchor_wr::Union{Nothing, Float64}` — Candidate's decided win rate against random anchor (or `nothing`)
- `promotion_score::Float64` — Composite score that triggered the promotion gate
- `gap_since_last::Int` — Iterations elapsed since the previous promotion
- `total_promotions_at_event::Int` — Total promotions count at the time of this event
- `elo_candidate::Float64`, `elo_best::Float64` — Elo ratings at promotion time
- `timestamp::String` — ISO 8601 UTC timestamp of the promotion event
"""
struct PromotionRecord
    iteration::Int
    win_rate_vs_best::Float64
    wins::Int
    losses::Int
    draws::Int
    random_anchor_wr::Union{Nothing, Float64}
    promotion_score::Float64
    gap_since_last::Int
    total_promotions_at_event::Int
    elo_candidate::Float64
    elo_best::Float64
    timestamp::String
end

# ── Progress Tracker ─────────────────────────
"""
    ProgressTracker

Tracks training progress state across iterations: promotion events, streaks, and gaps.

# Fields
- `last_best_iter::Int` — Iteration of the most recent promotion (0 if none)
- `total_promotions::Int` — Running count of promotion events
- `inter_promotion_gaps::Vector{Int}` — Iteration gaps between consecutive promotions
- `longest_streak::Int` — Longest streak without a promotion
- `current_streak::Int` — Current ongoing streak without a promotion
- `promotions::Vector{PromotionRecord}` — History of all promotion records
"""
mutable struct ProgressTracker
    last_best_iter::Int
    total_promotions::Int
    inter_promotion_gaps::Vector{Int}
    longest_streak::Int
    current_streak::Int
    promotions::Vector{PromotionRecord}
end

"""
    ProgressTracker() -> ProgressTracker

Construct an empty ProgressTracker with zero counts and no promotion history.
"""
function ProgressTracker()
    return ProgressTracker(0, 0, Int[], 0, 0, PromotionRecord[])
end

"""
    record_promotion!(pt::ProgressTracker, iter::Int, record::PromotionRecord)

Record a promotion event: stores the promotion record, updates the gap from the previous
promotion, increments the counter, and resets the current streak.
"""
function record_promotion!(pt::ProgressTracker, iter::Int, record::PromotionRecord)
    if pt.last_best_iter > 0
        gap = iter - pt.last_best_iter
        push!(pt.inter_promotion_gaps, gap)
    end
    pt.total_promotions += 1
    pt.last_best_iter = iter
    pt.current_streak = 0
    push!(pt.promotions, record)
end

"""
    record_non_promotion!(pt::ProgressTracker)

Record a non-promotion iteration: increments the current streak and updates the
longest streak if the current streak exceeds it.
"""
function record_non_promotion!(pt::ProgressTracker)
    pt.current_streak += 1
    if pt.current_streak > pt.longest_streak
        pt.longest_streak = pt.current_streak
    end
end

# ── Learning Curve CSV ───────────────────────
"""
    CSV_HEADER

Learning curve CSV column header. Written once per file via `write_csv_header`.
Columns: iteration, avg_loss, policy_loss, value_loss, grad_norm, pred_entropy,
target_entropy, param_update_norm, replay_fill_pct, avg_game_len, baseline_wr,
candidate_vs_best_wr, promoted, elo_candidate, elo_best.
"""
const CSV_HEADER = "iteration,avg_loss,policy_loss,value_loss,grad_norm,pred_entropy,target_entropy,param_update_norm,replay_fill_pct,avg_game_len,baseline_wr,candidate_vs_best_wr,promoted,elo_candidate,elo_best"

"""
    write_csv_header(io::IO)

Write the CSV header line to `io`. Idempotent — caller decides whether to call
on fresh vs resumed files.
"""
function write_csv_header(io::IO)
    println(io, CSV_HEADER)
end

"""
    write_csv_row(io::IO, iter, loss, policy_loss, value_loss, grad_norm, pred_ent,
                  target_ent, param_norm, replay_pct, game_len, baseline_wr,
                  candidate_wr, promoted, elo_candidate, elo_best)

Write a single iteration row to the learning curve CSV file.
All numeric values are formatted with `"%.6f"`; `candidate_wr` may be `nothing`
(emits an empty cell) on bootstrap iterations where no best model exists yet.
"""
function write_csv_row(io::IO, iter::Int, loss, policy_loss, value_loss, grad_norm, pred_ent, target_ent, param_norm, replay_pct, game_len, baseline_wr, candidate_wr, promoted::Bool, elo_candidate::Float64, elo_best::Float64)
    println(io, join(
        [string(iter),
         format_float(loss), format_float(policy_loss), format_float(value_loss),
         format_float(grad_norm), format_float(pred_ent), format_float(target_ent),
         format_float(param_norm), format_float(replay_pct), format_float(game_len),
         format_float(baseline_wr), format_float_or_empty(candidate_wr),
         promoted ? "yes" : "no",
         format_float(elo_candidate), format_float(elo_best)],
        ","
    ))
end

"""Format a value as a 6-decimal string for CSV output."""
format_float(x) = @sprintf("%.6f", Float64(x))

"""Return an empty string for `nothing` (CSV blank cell), otherwise delegate to `format_float`."""
format_float_or_empty(x::Nothing) = ""
format_float_or_empty(x::Real) = format_float(x)

# ── Diagnostics block ────────────────────────
"""
    print_progress_diagnostics(param_update_norm, pt::ProgressTracker, elo::EloTracker, current_iter::Int)

Print a formatted training-progress diagnostics block: parameter update norm,
current best iteration, iterations since last promotion, total promotions,
average gap between promotions, longest streak without promotion, and Elo ratings.
"""
function print_progress_diagnostics(param_update_norm::Real, pt::ProgressTracker, elo::EloTracker, current_iter::Int)
    println("  ── Training Progress ─────────────────────────")
    println("    Param update norm         : $(round(param_update_norm, digits=6))")
    println("    Current best iteration    : $(pt.last_best_iter)")
    iters_since = pt.last_best_iter > 0 ? current_iter - pt.last_best_iter : current_iter
    println("    Iterations since promotion: $(iters_since)")
    println("    Total promotions          : $(pt.total_promotions)")
    if pt.total_promotions >= 2
        avg_gap = sum(pt.inter_promotion_gaps) / length(pt.inter_promotion_gaps)
        println("    Avg gap between promos    : $(round(avg_gap, digits=1))")
    end
    println("    Longest streak no promo   : $(pt.longest_streak)")
    println("    Elo candidate / best      : $(round(Int, elo.candidate_rating)) / $(round(Int, elo.best_rating))")
    println("  ────────────────────────────────────────────────")
end

"""
    print_promotion_event(io, promotion_num, gap, win_rate)

Print a formatted promotion event message.
"""
function print_promotion_event(promotion_num::Int, gap::Int, win_rate::Union{Nothing, Float64})
    println("  ── Promotion #$(promotion_num) ────────────────────────")
    println("    Iterations since previous promotion: $(gap)")
    if win_rate !== nothing
        println("    Win rate vs previous best: $(round(win_rate, digits=1))%")
    end
    println("  ────────────────────────────────────────────────")
end

"""
    compute_value_calibration(v_pred, v_target)

Compute value-head calibration metrics between predicted values `v_pred` and
actual returns `v_target`. Returns a NamedTuple with `mae`, `pearson_r`, and
`spearman_rho`. Returns `(NaN, NaN, NaN)` when either input is degenerate
(all values identical).
"""
function compute_value_calibration(v_pred::AbstractVector{Float32}, v_target::AbstractVector{Float32})
    if std(v_pred) == 0.0f0 || std(v_target) == 0.0f0
        return (mae=NaN, pearson_r=NaN, spearman_rho=NaN)
    end

    mae = mean(abs.(v_pred .- v_target))
    pearson_r = cor(v_pred, v_target)

    # Spearman ρ: convert to ranks, then compute Pearson correlation on ranks
    n = length(v_pred)
    pred_ranks = zeros(Float64, n)
    target_ranks = zeros(Float64, n)
    pred_ranks[sortperm(v_pred)] .= Float64.(1:n)
    target_ranks[sortperm(v_target)] .= Float64.(1:n)
    spearman_rho = cor(pred_ranks, target_ranks)

    return (mae=mae, pearson_r=pearson_r, spearman_rho=spearman_rho)
end

"""
    print_historical_summary(pt::ProgressTracker)

Print a formatted summary of all promotion history recorded in the progress tracker.
"""
function print_historical_summary(pt::ProgressTracker)
    println("  ── Promotion History Summary ──────────────────")
    println("    Total promotions          : $(pt.total_promotions)")

    if !isempty(pt.promotions)
        win_rates = [p.win_rate_vs_best for p in pt.promotions]
        avg_wr = sum(win_rates) / length(win_rates)
        println("    Avg win rate              : $(round(avg_wr, digits=1))%")

        sorted_wr = sort(win_rates)
        n_wr = length(sorted_wr)
        median_wr = n_wr % 2 == 1 ? sorted_wr[(n_wr + 1) ÷ 2] : (sorted_wr[n_wr ÷ 2] + sorted_wr[n_wr ÷ 2 + 1]) / 2
        println("    Median win rate           : $(round(median_wr, digits=1))%")

        best_idx = argmax(win_rates)
        println("    Best promotion            : $(round(win_rates[best_idx], digits=1))% (iter $(pt.promotions[best_idx].iteration))")

        worst_idx = argmin(win_rates)
        println("    Closest promotion         : $(round(win_rates[worst_idx], digits=1))% (iter $(pt.promotions[worst_idx].iteration))")
    end

    if !isempty(pt.inter_promotion_gaps)
        avg_gap = sum(pt.inter_promotion_gaps) / length(pt.inter_promotion_gaps)
        println("    Avg gap between promos    : $(round(avg_gap, digits=1)) iters")

        sorted_gaps = sort(pt.inter_promotion_gaps)
        n_gap = length(sorted_gaps)
        median_gap = n_gap % 2 == 1 ? Float64(sorted_gaps[(n_gap + 1) ÷ 2]) : (sorted_gaps[n_gap ÷ 2] + sorted_gaps[n_gap ÷ 2 + 1]) / 2.0
        println("    Median gap                : $(round(median_gap, digits=1)) iters")
    end

    println("    Longest streak no promo   : $(pt.longest_streak) iters")
    println("  ────────────────────────────────────────────────")
end

end # module
