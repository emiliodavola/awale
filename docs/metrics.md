# Training Metrics Reference

> Formal documentation of every metric emitted by the training loop:
> mathematical definition, code location, interpretation, and computational cost.
>
> **Metric version**: 1.0.0
> **Last updated**: 2026-07-29

## Legend

| Symbol | Meaning |
|--------|---------|
| `B` | Batch size (positions per gradient update) |
| `U` | Updates per iteration |
| `N` | Self-play positions generated per iteration |
| `T` | Training iterations elapsed |

All metrics are **passive observations** — they never modify training behavior, promotion logic, MCTS, or hyperparameters.

---

## 1. Core Losses

### Combined Loss

$$L = L_{\text{policy}} + L_{\text{value}}$$

| Property | Value |
|----------|-------|
| **Source** | `Training.train_step` → `step_result.loss` |
| **Formula** | Sum of policy cross-entropy and value MSE |
| **Reuse** | Already computed for backpropagation |
| **Cost** | Time: O(B·U) (free — gradient computation dominates); Mem: 4 bytes/iter |

### Policy Loss

$$L_{\text{policy}} = -\frac{1}{B}\sum_{i=1}^{B} \sum_{a} \pi_{\text{MCTS}}(a|s_i) \cdot \log \pi_{\theta}(a|s_i)$$

Cross-entropy between the MCTS target policy and the network's predicted policy.

| Property | Value |
|----------|-------|
| **Source** | `Training.train_step` → `step_result.policy_loss` |
| **Expected range** | [0.5, 3.0] early training, converges toward [0.1, 1.0] |
| **High** | Network policy diverges from MCTS guidance |
| **Low** | Network closely matches MCTS — possible policy convergence |
| **Cost** | Free (reuses loss computation) |

### Value Loss

$$L_{\text{value}} = \frac{1}{B}\sum_{i=1}^{B} (v_\theta(s_i) - z_i)^2$$

Mean squared error between the network's value prediction and the actual game outcome (from backfilled self-play returns).

| Property | Value |
|----------|-------|
| **Source** | `Training.train_step` → `step_result.value_loss` |
| **Expected range** | [0.5, 2.0] early, converges toward [0.1, 0.8] |
| **High** | Value head poorly calibrated — expected early in training |
| **Low** | Value head predicts outcomes accurately |
| **Cost** | Free (reuses loss computation) |

---

## 2. Gradient & Parameter Metrics

### Gradient Norm

$$||\nabla_\theta L||_2 = \sqrt{\sum_{i} g_i^2}$$

L2 norm of the gradient vector across all model parameters.

| Property | Value |
|----------|-------|
| **Source** | `Training.train_step` → `step_result.grad_norm` |
| **Expected range** | [0.1, 10.0] depending on model size and loss scale |
| **High** | Large updates — may indicate instability or loss spikes |
| **Low** | Small gradients — may indicate saturation or convergence |
| **Cost** | Free (reuses gradient from backprop) |

### Parameter Update Norm

$$||\Delta\theta||_2 = ||\theta_{\text{new}} - \theta_{\text{old}}||_2$$

L2 norm of the parameter change vector before and after the training iteration's gradient updates. Captures ALL parameters: weights, biases, embeddings; excludes BatchNorm if present.

| Property | Value |
|----------|-------|
| **Source** | `train.jl` — `Flux.destructure` before/after `run_training_iteration` |
| **Expected range** | [0.001, 0.5] for stable training |
| **High** | > 1.0 may indicate instability or a large policy change |
| **Low** | < 0.001 suggests minimal learning |
| **Cost** | O(P) where P = parameter count (~100K–300K); Mem: P × 4 bytes (shared with drift) |

**Reuse**: The `destructure` vector is also used for network drift — no additional capture call needed (see [R1 modification](openspec/changes/training-observability/specs/training-metrics/spec.md#r1--parameter-update-norm-computation-reuse)).

---

## 3. Policy Entropy

### Target (MCTS) Policy Entropy

$$H(\pi_{\text{MCTS}}) = -\sum_a \pi_{\text{MCTS}}(a|s) \cdot \log \pi_{\text{MCTS}}(a|s)$$

| Property | Value |
|----------|-------|
| **Source** | `Training.train_step` → `step_result.target_entropy` |
| **Expected range** | [0.0, 1.8] for 6-action domain |
| **High** | MCTS policy is diffuse — many moves considered plausible |
| **Low (≈ 0)** | MCTS strongly favours a single move — near-deterministic |
| **Cost** | Free (reuses target policy from loss computation) |

### Predicted (Network) Policy Entropy

$$H(\pi_\theta) = -\sum_a \pi_\theta(a|s) \cdot \log \pi_\theta(a|s)$$

| Property | Value |
|----------|-------|
| **Source** | `Training.train_step` → `step_result.pred_entropy` |
| **Expected range** | [0.0, 1.8] |
| **Typically** | Slightly higher than MCTS entropy — network produces softer distributions |
| **Cost** | Free (reuses predicted policy from loss computation) |

---

## 4. MCTS vs Network Agreement

### KL Divergence

$$D_{\text{KL}}(\pi_{\text{MCTS}} || \pi_\theta) = \sum_a \pi_{\text{MCTS}}(a|s) \cdot \log\frac{\pi_{\text{MCTS}}(a|s)}{\pi_\theta(a|s)}$$

Direction: KL(target || predicted). We use this direction because $\pi_{\text{MCTS}}$ is the teaching signal — the divergence measures how much information the network has yet to learn from the search. The reverse direction $D_{\text{KL}}(\pi_\theta || \pi_{\text{MCTS}})$ would be dominated by regions where the network assigns probability but MCTS does not, which is less informative for learning progress.

| Property | Value |
|----------|-------|
| **Source** | `Training.run_training_iteration` → `TrainingResult.kl_mean` (also median, P25, P75, P95 in console) |
| **Expected range** | Mean: [0.05, 0.5] for active learning; < 0.05 suggests policy convergence |
| **Observed** | ~0.12 at ~60% Top-1 (healthy search contribution) |
| **High** | Network policy differs significantly from MCTS — search still provides new information |
| **Low (→ 0)** | Both policies nearly identical — search no longer improves policy |
| **Cost** | Free: computed from existing logits and target policy; no forward pass needed |

### Top-K Agreement

$$\text{Top-}K = \frac{1}{B}\sum_{i=1}^{B} [\text{argmax}_K(\pi_{\text{MCTS}}(s_i)) \cap \text{argmax}_K(\pi_\theta(s_i)) \neq \emptyset]$$

Proportion of positions where the network's top-K actions include the MCTS's top choice.

| Property | Value |
|----------|-------|
| **Source** | `Training.run_training_iteration` → `TrainingResult.top1_pct` (also top2, top3) |
| **Expected** | Top-1: 40–80% during active learning; > 95% → search may no longer improve policy |
| **Observed** | Top-1 ≈ 60% (healthy exchange) |
| **Cost** | Free: computed from existing logits; O(B·A·log A) sorting |

### Policy Distance (L1)

$$L_1 = \frac{1}{2B}\sum_{i=1}^{B} \sum_a |\pi_{\text{MCTS}}(a|s_i) - \pi_\theta(a|s_i)|$$

Scaled by 1/2 so the range is [0, 1] (full range of a probability difference).

| Property | Value |
|----------|-------|
| **Source** | `Training.run_training_iteration` → `TrainingResult.l1_mean` |
| **Expected range** | [0.05, 0.35] during active learning |
| **Observed** | ~0.18 (consistent with KL ≈ 0.12) |
| **High** | → 0.5 means policies disagree substantially |
| **Low** | → 0 means policies are identical |
| **Cost** | Free: computed from existing predictions |

### Root Confidence

$$\text{RootConf} = \frac{\max_a N(a)}{\sum_a N(a)}$$

Ratio of the most-visited action's visit count to total visits at the MCTS root node. Measures how decisive the search is.

| Property | Value |
|----------|-------|
| **Source** | `MCTS.search_with_stats` (3rd return), accumulated in `all_root_confidences` |
| **Expected range** | [0.2, 1.0]; ~0.55 mean is typical for a domain with multiple competitive moves |
| **High (→ 1)** | MCTS strongly favours one move — position likely has a clear best answer |
| **Low (→ 0.2)** | Multiple moves appear equally promising |
| **Cost** | Free: visit counts already computed by search |

### MCTS Root Q

$$Q_{\text{root}} = \frac{\sum_{n \in \text{children}} v_n \cdot N_n}{V_{\text{root}}}$$

The value at the root node after search, computed as mean value across all visited children. Used to measure how much the search improves over the raw network evaluation.

| Property | Value |
|----------|-------|
| **Source** | `MCTS.search_with_stats` (4th return — `root.value_sum / root.visits`) |
| **Expected range** | [-1, 1] |
| **Cost** | O(1): already computed during search (simply a read of accumulated value) |

---

## 5. Network Change Metrics

### Network Drift

$$D_{\text{drift}} = \frac{1}{|\mathcal{R}|}\sum_{s \in \mathcal{R}} D_{\text{KL}}(\pi_{\theta_{\text{before}}}(s) || \pi_{\theta_{\text{after}}}(s))$$

KL divergence between the network's policy before and after a training iteration, measured over a **fixed reference set** $\mathcal{R}$ of 200 game states (generated at startup with a fixed seed). The set is fixed so drift measurements are comparable across iterations.

| Property | Value |
|----------|-------|
| **Source** | `train.jl` — computed from reference set before/after `run_training_iteration` |
| **Expected range** | [0.001, 0.02] for stable training |
| **Observed** | ~0.003–0.007 (very stable) |
| **High** | > 0.05 suggests the policy is changing rapidly — possible instability |
| **Low (→ 0)** | Model unchanging — may have converged |
| **Cost** | O(R·A) where R = 200 (reference set size) — one forward pass per iteration; Mem: ~200 × 6 × 4 bytes |

### Drift Convergence Warning

When drift KL ≈ 0 (< ε = 0.0001 variance over 20 iterations), training may have converged. Printed as:

```
⚠ Network drift near zero — training may have converged
```

---

## 6. Value Calibration

### MAE

$$\text{MAE} = \frac{1}{B}\sum_{i=1}^{B} |v_\theta(s_i) - z_i|$$

### Pearson Correlation

$$r = \frac{\sum (v_i - \bar{v})(z_i - \bar{z})}{\sqrt{\sum (v_i - \bar{v})^2 \sum (z_i - \bar{z})^2}}$$

### Spearman Rank Correlation

$$\rho = r(\text{rank}(v), \text{rank}(z))$$

Pearson correlation applied to ranked predictions and targets.

| Property | Value |
|----------|-------|
| **Source** | `Metrics.compute_value_calibration(v_pred, v_target)` |
| **Expected (MAE)** | [0.1, 1.0]; decreases as value head improves |
| **Expected (Pearson)** | [0.3, 0.9]; positive means value head correlates with outcomes |
| **Expected (Spearman)** | [0.3, 0.9]; rank-order agreement |
| **Cost** | Time: O(B log B) for ranking; Mem: O(B) — reuses last batch's predictions |

No extra forward passes — computed from the last training batch's value outputs.

---

## 7. Elo System

$$E_{\text{candidate}} = \frac{1}{1 + 10^{(\text{rating}_{\text{best}} - \text{rating}_{\text{candidate}}) / 400}}$$

$$S_{\text{candidate}} = \frac{W + 0.5D}{W + L + D}$$

$$\Delta = K \cdot (S - E)$$

| Property | Value |
|----------|-------|
| **Source** | `Metrics.EloTracker`, computed in `train.jl` |
| **K (default)** | 64 (was 32 — increased for faster diagnostic signal) |
| **Expected delta** | ±1–10 per iteration depending on WR gap |
| **Upset** | $|S - E|$ — how far the result deviated from expectation |
| **Cost** | O(1): trivial arithmetic |

---

## 8. Δ-Metrics (Trends)

| Metric | Formula | NaN condition |
|--------|---------|--------------|
| ΔKL | $|KL_i - KL_{i-1}|$ | First iteration |
| ΔPolicy distance | $|L1_i - L1_{i-1}|$ | First iteration |
| ΔTop1 | $|\text{Top1}_i - \text{Top1}_{i-1}|$ | First iteration |
| ΔDrift | $|\text{Drift}_i - \text{Drift}_{i-1}|$ | First iteration |
| ΔGrad norm | $|\text{Grad}_i - \text{Grad}_{i-1}|$ | First iteration |
| ΔParam update | $|\Delta\theta_i - \Delta\theta_{i-1}|$ | First iteration |

**Cost**: O(1): difference between stored scalar values.

---

## 9. Moving Averages

Rolling mean over windows of 5, 10, and 20 iterations for:

- `avg_loss`, `policy_loss`, `value_loss`
- `kl_mean`, `top1_pct`
- `drift_kl`
- `elo_candidate`
- `pred_entropy`

NaN until the window is fully populated (e.g., MA-20 → NaN for iterations 1–19).

**Cost**: O(W) per metric where W = window size. Ring buffer memory: W × 4 bytes.

---

## 10. Replay Fill %

$$\text{Replay Fill \%} = \frac{|\text{ReplayBuffer}|}{\text{Capacity}} \times 100$$

Percentage of the circular replay buffer currently occupied.

| Property | Value |
|----------|-------|
| **Source** | `Training.run_training_iteration` (line 342) |
| **Former name** | "Replay Coverage" (renamed because it measures buffer fill, not state-space coverage) |
| **Expected range** | 0% → grows monotonically → 100% at capacity |
| **Cost** | O(1): `length(buffer) / capacity` |

**Important**: This is NOT the fraction of the state space explored, nor the fraction of samples consumed. It is simply how full the circular buffer is.

---

## 11. Diagnostic Dashboard

### Convergence Detection

Sliding window (20 iterations) checks variance of KL, drift, Top1, and param update magnitude against thresholds:

| Signal | Threshold ε | State |
|--------|------------|-------|
| KL variance | 0.001 | ACTIVE if var > ε, STALLED otherwise |
| Drift variance | 0.0001 | ACTIVE if var > ε, STALLED otherwise |
| Top1 variance | 1.0 (%) | ACTIVE if var > ε, STALLED otherwise |
| Param update norm | 0.01 | ACTIVE if > ε, STALLED otherwise |

### Health Dashboard

Single-line assessment per iteration:

```
Net:ACTIVE Srch:HIGH Drift:LOW Promo:1/50 Rply:84.8% ValCal:OK
```

| Component | Heuristic |
|-----------|-----------|
| **Net** (Network learning) | ACTIVE if ΔKL > 0.01 or param_update_norm > 0.01; else STALLED |
| **Srch** (Search usefulness) | HIGH if Top1 < 80%; LOW otherwise |
| **Drift** | LOW if drift_kl < 0.01; MEDIUM if < 0.05; HIGH otherwise |
| **Promo** | iters_since_promotion / total_promotions |
| **Rply** | replay_fill_pct |
| **ValCal** | OK if value calibration data available; N/A otherwise |

### Diagnostic Warnings

| Trigger | Message |
|---------|---------|
| Top-1 > 95% | `⚠ Top-1 agreement N% > 95% — search may no longer improve policy` |
| Drift variance ≈ 0 | `⚠ Network drift near zero — training may have converged` |
| Param update > 5× rolling mean | `⚠ Parameter update unusually large — possible instability` |

All warnings are purely informational — no circuit breakers or behavioral changes.

---

## 12. Cost Summary

| Metric | Time Cost | Mem Cost | Frequency | Forward Passes |
|--------|-----------|----------|-----------|----------------|
| Policy/Value loss | Free (loss) | 4 B | Every iter | 0 (reused) |
| Gradient norm | Free (gradient) | 4 B | Every iter | 0 (reused) |
| Entropies | Free (loss) | 8 B | Every iter | 0 (reused) |
| KL divergence | O(B·A) | ∼B·4 B | Every iter | 0 (reused) |
| Top-K agreement | O(B·A·log A) | ∼B·3 B | Every iter | 0 (reused) |
| Policy L1 | O(B·A) | ∼B·4 B | Every iter | 0 (reused) |
| Root confidence | O(1) | 4 B/pos | Every pos | 0 (MCTS) |
| Root Q | O(1) | 4 B | Every iter | 0 (MCTS) |
| Network drift | O(R·A) | R·A·4 B | Every iter | 1 forward (R=200) |
| Param update norm | O(P) | P·4 B | Every iter | 0 (destructure) |
| Value calibration | O(B log B) | ∼B·8 B | Every iter | 0 (reused) |
| Elo | O(1) | 24 B | Every iter | 0 |
| Δ-metrics | O(1) | 32 B | Every iter | 0 |
| Moving averages | O(W) per metric | W·4 B each | Every iter | 0 |
| Convergence | O(W) | W·4 B | Every iter | 0 |
| Health dashboard | O(1) | 0 | Every iter | 0 |
| Warnings | O(1) | 0 | Every iter | 0 |
| JSONL write | O(F) ∼ 1 KB | 1 KB flush buf | Every iter | 0 |

Where: B = batch size, A = 6 (actions), P = parameter count, R = 200 (reference set), W = 20 (window).

Total overhead per iteration: ~1 KB JSONL write + 1 forward pass for drift (R=200 states). No other forward passes added.
