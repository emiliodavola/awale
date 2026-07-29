"""
    Metrics

Training-progress tracking: Elo ratings, promotion history, learning curve CSV,
and inter-iteration diagnostics logging.
"""
module Metrics

using ..State: GameState
using ..Model: encode_state
using Dates
using Printf

export EloTracker, ProgressTracker, PromotionRecord, update_elo!, promote_elo!,
       write_csv_header, write_csv_row, print_progress_diagnostics, print_promotion_event

# ── Elo ──────────────────────────────────────
mutable struct EloTracker
    candidate_rating::Float64
    best_rating::Float64
    k::Float64
    history::Vector{Tuple{Int, Float64, Float64}}
end

function EloTracker(; candidate=1500.0, best=1500.0, k=32.0)
    return EloTracker(candidate, best, k, Tuple{Int, Float64, Float64}[])
end

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
struct PromotionRecord
    iteration::Int
    win_rate_vs_best::Float64
    wins::Int
    losses::Int
    draws::Int
    random_anchor_wr::Union{Nothing, Float64}
    promotion_score::Float64
end

# ── Progress Tracker ─────────────────────────
mutable struct ProgressTracker
    last_best_iter::Int
    total_promotions::Int
    inter_promotion_gaps::Vector{Int}
    longest_streak::Int
    current_streak::Int
    promotions::Vector{PromotionRecord}
end

function ProgressTracker()
    return ProgressTracker(0, 0, Int[], 0, 0, PromotionRecord[])
end

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

function record_non_promotion!(pt::ProgressTracker)
    pt.current_streak += 1
    if pt.current_streak > pt.longest_streak
        pt.longest_streak = pt.current_streak
    end
end

# ── Learning Curve CSV ───────────────────────
const CSV_HEADER = "iteration,avg_loss,policy_loss,value_loss,grad_norm,pred_entropy,target_entropy,param_update_norm,replay_fill_pct,avg_game_len,baseline_wr,candidate_vs_best_wr,promoted,elo_candidate,elo_best"

function write_csv_header(io::IO)
    println(io, CSV_HEADER)
end

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

format_float(x) = @sprintf("%.6f", Float64(x))
format_float_or_empty(x::Nothing) = ""
format_float_or_empty(x::Real) = format_float(x)

# ── Diagnostics block ────────────────────────
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

end # module
