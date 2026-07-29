# MCTS Diagnostics Specification

## Purpose

Add 7 telemetry metrics that quantify MCTS vs network policy alignment inside each training iteration, with zero algorithm or hyperparameter changes. All metrics are read-only: they observe, never modify, the training loop.

## Requirements

### R1: KL Divergence

The system MUST compute the per-position KL divergence between the MCTS target policy (`pi_target` / `Y_pi`) and the network's post-update policy (`pi_network` / `after_probs`):

`KL(pi_target || pi_network) = sum(pi_target * log(pi_target / pi_network))`

**Constraints:**
- `pi_network` MUST be clamped to `max(after_probs, 1e-10)` before the log to prevent `-Inf`.
- Computation MUST reuse `Y_pi` and `after_probs` already computed in `train_step` (zero new forward passes).
- Stored values MUST be combined across all batches in one iteration.
- Report MUST include: mean, median, max, P25, P75, P95.

### R2: Top-K Agreement

The system MUST compute the percentage of positions where `argmax(pi_target)` and `argmax(pi_network)` agree.

**Constraints:**
- MUST compute **Top-1**, **Top-2**, and **Top-3** agreement rates.
- Top-K agreement: `1` if `action_target in top_k(network_actions)`, `0` otherwise, averaged over all positions.
- MUST reuse `Y_pi` and `after_probs` from `train_step`.
- Report MUST show percentages formatted to one decimal place (e.g. `XX.X%`).

### R3: Root Confidence

The system MUST accumulate the maximum visit-count probability from each game's MCTS root policy.

**Constraints:**
- The root visit-count policy is already returned by `search_with_stats` in `collect_selfplay_data`.
- MUST accumulate one `max(counts)` value per game (not per position).
- Report MUST include: mean, median, P25, P50, P75, P95, min, max.

### R4: Target Policy Entropy (Enhanced)

The system MUST enhance the existing `target_entropy` reporting with distributional statistics.

**Constraints:**
- `target_entropy = -sum(Y_pi * log(clamp(Y_pi, 1e-10, 1.0)))` already computed in `train_step`.
- MUST accumulate per-sample (not batch-averaged) values across all batches.
- Report MUST include: mean, median, min, max, P25, P75, P95.

### R5: Policy Distance (L1)

The system MUST compute the normalized L1 distance between `pi_network` and `pi_target`:

`L1 = sum(|pi_network - pi_target|) / 2`

**Constraints:**
- Normalization factor of `2` ensures the metric is in `[0, 1]`.
- MUST reuse `after_probs` and `Y_pi` from `train_step`.
- Report MUST include: mean, median, P25, P75, P95.

### R7: Network Drift

The system MUST measure how far the network's policy has drifted from its state at iteration 1, computed over a fixed reference state set.

**Reference set:**
- MUST contain ~200 canonicalized states sampled from the self-play games of iteration 1.
- MUST be sampled once at training start and persisted for the entire run.
- Each state is a `GameState` object.

**Computation:**
- At iteration 1: run `predict_batch_inference` on the reference set, store reference `logits_ref`.
- After each iteration's training: run `predict_batch_inference` on the same reference set to get `logits_curr`.
- Compute `KL(softmax(logits_ref) || softmax(logits_curr))`, averaged over all reference states.
- Report a single scalar per iteration.

**Constraints:**
- Exactly ONE extra `predict_batch_inference` call per iteration (plus one at iteration 1 for the reference).
- `predict_batch_inference` MUST run in inference mode (`Flux.testmode!`).
- Clamp current softmax probabilities to `1e-10` before the log.

### R8: Output Format

The system MUST print a dedicated diagnostics block after the existing diagnostics in `run_training_iteration`:

```
──────────────────────────────────────────────
MCTS Diagnostics
Avg KL(target || network)   Median   Max   P25 / P75 / P95
Top-1 agreement: XX.X%   Top-2: XX.X%   Top-3: XX.X%
Root confidence — Mean / Median / P25 / P50 / P75 / P95 / Min / Max
Policy distance (L1) — Mean / Median / P25 / P75 / P95
──────────────────────────────────────────────
```

And in `train.jl`, after the training loop:

```
──────────────────────────────────────────────
Network Drift: X.XXXX
──────────────────────────────────────────────
```

### R9: Percentile Definition

All percentiles (P25, P50, P75, P95) MUST use linear interpolation between sorted values, matching Julia's `Statistics.quantile` default.

## Constraints

### C1: Zero New Forward Passes (Metrics 1-5)

Metrics 1-5 MUST derive entirely from values already computed inside `train_step` (`after_probs`, `Y_pi`, `after_log_probs`). No additional calls to `predict_raw`, `predict_inference`, or `predict_batch_inference`.

### C2: Single Extra Forward Pass (Metric 7)

Network Drift SHALL add exactly ONE `predict_batch_inference` call per iteration (the reference set pass). The reference set generation at iteration 1 uses the same call — it is NOT an extra call on top of the per-iteration budget.

### C3: No Algorithm Changes

The system MUST NOT modify `MCTS.jl`, `Model.jl`, `ReplayBuffers.jl`, `Env.jl`, or `State.jl`. All behavior, hyperparameters, and search logic MUST remain identical.

### C4: Numerical Determinism

Adding diagnostics MUST NOT change the training loop's numerical outputs. Loss values, gradients, and model weights MUST be bit-identical with the baseline.

## Acceptance Criteria

### A1: All 7 metrics printed each iteration

- [ ] KL divergence block appears after existing Diagnostics.
- [ ] Network Drift line prints in `train.jl` after training.

### A2: Bit-identical training

- [ ] `train_step` returns identical loss tuple with and without diagnostics.
- [ ] Existing test suite passes unchanged.

### A3: Percentile correctness

- [ ] P50 == median for metrics with sorted inputs.
- [ ] P25 <= P50 <= P75 <= P95 always.

### A4: No module contamination

- [ ] `git diff --stat` shows changes only in `src/Awale/Training.jl` and `train.jl`.

## Test Scenarios

### S1: KL baseline match

- GIVEN a known `pi_target` and `pi_network`
- WHEN KL is computed
- THEN the result matches a reference calculation using `sum(pi_target .* log.(pi_target ./ max.(pi_network, 1e-10)))`

### S2: Top-K counts

- GIVEN 100 positions where argmax matches in 73
- WHEN Top-1 agreement is computed
- THEN result is `73.0%`

### S3: L1 bounds

- GIVEN identical `pi_target == pi_network`
- WHEN L1 is computed
- THEN result is `0.0`
- GIVEN one-hot at index 1 vs one-hot at index 2
- THEN result is `1.0`

### S4: Root confidence range

- GIVEN any game's root visit-count policy
- WHEN max(counts) is taken
- THEN value is in `[0.0, 1.0]`

### S5: Network drift stability

- GIVEN a fixed reference set and identical model weights
- WHEN Network Drift is computed twice
- THEN results are identical (deterministic)
