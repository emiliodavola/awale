# Tasks: Intra-Iteration MCTS Diagnostics

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | ~150–200 |
| 400-line budget risk | Low |
| Chained PRs recommended | No |
| Suggested split | Single PR |
| Delivery strategy | single-pr |
| Chain strategy | size-exception |

Decision needed before apply: No
Chained PRs recommended: No
Chain strategy: size-exception
400-line budget risk: Low

### Suggested Work Units

| Unit | Goal | Likely PR | Notes |
|------|------|-----------|-------|
| 1 | All metrics + drift in one PR | PR 1 | base=main; all 7 metrics, accumulator, diagnostics block, network drift |

## Phase 1: Foundation — Return `after_probs` and `Y_pi` from `train_step`

- [x] 1.1 Add `after_probs` and `Y_pi` to `train_step`'s named tuple return (`src/Awale/Training.jl` ~110–130) — both are already computed inside the function, zero new computation
- [x] 1.2 Update `run_training_iteration` destructuring at line 247 to capture `after_probs` and `Y_pi`

## Phase 2: Stats Accumulator + Per-Sample Metrics

- [x] 2.1 Add `MCTSDiagnostics` mutable struct with `Vector{Float32}` accumulators for: `kl_divs`, `l1_dists`, `target_entropies`, `root_confidences` (`src/Awale/Training.jl`, after line 16)
- [x] 2.2 Compute per-position KL divergence from `Y_pi` and `after_probs` inside the training loop — KL = sum(Y_pi .* log.(max.(after_probs, 1e-10f0)), dims=1), push each value to `kl_divs`
- [x] 2.3 Compute per-position Top-1/2/3 agreement — argmax check for Top-1, `topk` for Top-2/3 — count matches, accumulate numerators and denominator
- [x] 2.4 Compute per-position L1 distance: `sum(abs.(after_probs - Y_pi), dims=1) / 2`, push to `l1_dists`
- [x] 2.5 Compute per-sample target entropy: `-sum(Y_pi .* log.(clamp.(Y_pi, 1e-10f0, 1.0f0)), dims=1)`, push to `target_entropies`

## Phase 3: Root Confidence

- [x] 3.1 After each game in `run_training_iteration` (line ~215), extract `max(first(game_data)[2])` — the max visit-count of the root position — and push to `root_confidences`

## Phase 4: MCTS Diagnostics Block

- [x] 4.1 Import `Statistics` at top of Training.jl module
- [x] 4.2 After existing Diagnostics block (after line ~279), compute stats with `Statistics.quantile` for each accumulator: mean, median, min, max, P25, P75, P95
- [x] 4.3 Print formatted MCTS Diagnostics block matching spec output format (KL, Top-K, root confidence, L1, entropy stats)
- [x] 4.4 Reset accumulators at start of each iteration to prevent cross-iteration leakage

## Phase 5: Network Drift in `train.jl`

- [x] 5.1 At iteration 1 (inside the training loop), after collecting self-play data but before training, extract ~200 `GameState` objects from the first game's data and persist as `ref_states::Vector{GameState}`
- [x] 5.2 Compute reference logits: `predict_batch_inference(model[], ref_states)` → `logits_ref`, store as `Vector{Float32}` of softmax probabilities
- [x] 5.3 After each iteration's training, compute current logits: `predict_batch_inference(model[], ref_states)` → `logits_curr`, compute `KL(softmax(logits_ref) || softmax(clamped_logits_curr))`
- [x] 5.4 After the training loop (after line 665), print Network Drift line: `Network Drift: X.XXXX`

## Phase 6: Determinism Verification

- [x] 6.1 Run training with and without diagnostics, confirm loss values and model weights are bit-identical
- [x] 6.2 Verify `train_step` returns identical loss tuple before and after the `after_probs`/`Y_pi` return addition

## Phase 7: Testing

- [x] 7.1 Unit test: KL divergence matches reference `sum(Y_pi .* log.(Y_pi ./ max.(after_probs, 1e-10f0)))` for known inputs
- [x] 7.2 Unit test: Top-1/2/3 agreement — 100 positions with 73 matches → `73.0%`
- [x] 7.3 Unit test: L1 bounds — identical policies → `0.0`, one-hot at 1 vs one-hot at 2 → `1.0`
- [x] 7.4 Unit test: Root confidence always in `[0.0, 1.0]`
- [x] 7.5 Unit test: P50 == median for any sorted accumulator
- [x] 7.6 Run full test suite: `julia --project=. -e 'using Pkg; Pkg.test()'`
