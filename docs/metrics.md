# Training Metrics Reference

> Formal documentation of every metric emitted by the training loop:
> mathematical definition, code location, interpretation, and computational cost.
>
> **Metric version**: 2.0.0
> **Last updated**: 2026-07-29

## Legend

| Symbol | Meaning |
|--------|---------|
| `B` | Batch size (positions per gradient update) |
| `U` | Updates per iteration |
| `N` | Self-play positions generated per iteration |
| `T` | Training iterations elapsed |
| `P` | Total model parameter count (~100K–300K) |
| `A` | Number of actions (6 in Awale) |
| `R` | Reference set size for drift (200) |
| `W` | Sliding window size for convergence (20) |

All metrics are **passive observations** — they never modify training behavior, promotion logic, MCTS, or hyperparameters.

**Log base convention**: All logarithms use $\ln$ (natural log) unless otherwise noted. Values are in nats. Elo uses $\log_{10}$.

---

## Metrics Classification

Metrics are organized into 7 categories by what aspect of training they measure:

| # | Category | Metrics | Purpose |
|---|----------|---------|---------|
| 1 | **Optimization** | Policy Loss, Value Loss, Gradient Norm, Parameter Update Norm | Core training loop health |
| 2 | **Policy Learning** | KL Divergence, Top-K Agreement, Policy Distance (L1), Target/ Predicted Policy Entropy | How well the network learns from MCTS |
| 3 | **Search** | Root Confidence, Root Q, Search Gain | MCTS search behaviour and contribution |
| 4 | **Value** | Value Loss (xref), Value Calibration (MAE, Pearson r, Spearman $\rho$) | Value head accuracy |
| 5 | **Replay** | Replay Fill % | Data pipeline state |
| 6 | **Promotion** | Win Rate, Elo Rating, Promotion History | Model improvement milestones |
| 7 | **Network Evolution** | Network Drift, Convergence Detection, Training Health Dashboard, Diagnostic Warnings | How the model changes over time |

---

## Metric Schema

Every metric entry follows this 8-field schema:

| Field | Content |
|-------|---------|
| **Definition** | Mathematical description — what the metric measures |
| **Formula** | LaTeX expression |
| **Variables** | Table of every symbol in the formula |
| **Implementation** | File, function, and line reference in source code |
| **Computational Cost** | Big-O cost per call and per iteration |
| **Interpretation** | What high/low values mean (universal, not project-specific) |
| **Typical Observations (this project)** | Empirical values seen during training runs — clearly labeled as observed, not expected |
| **Notes** | Edge cases, warnings, NaN conditions, and implementation gotchas |

---

## 1. Optimization Metrics

Measures from the gradient update loop — how well the loss decreases and how aggressively parameters change.

### Policy Loss

**Definition**

Cross-entropy between the MCTS target policy (teaching signal) and the network's predicted policy. This is the primary learning signal for the policy head.

**Formula**

$$L_{\text{policy}} = -\frac{1}{B}\sum_{i=1}^{B} \sum_{a} \pi_{\text{MCTS}}(a|s_i) \cdot \ln \pi_{\theta}(a|s_i)$$

**Variables**

| Variable | Description |
|----------|-------------|
| $B$ | Batch size |
| $\pi_{\text{MCTS}}(a\|s_i)$ | Target policy from MCTS for state $s_i$, action $a$ |
| $\pi_{\theta}(a\|s_i)$ | Network's predicted policy for state $s_i$, action $a$ |

**Implementation**

`src/Awale/Training.jl` — `train_step` function, line 195:
```julia
policy_loss = -sum(Y_pi .* after_log_probs) / size(X, 2)
```
`Y_pi` contains the MCTS target policies; `after_log_probs` contains $\ln \pi_{\theta}$ after the gradient update.

**Computational Cost**

- Per call: $O(B \cdot A)$ — element-wise multiplication and sum
- Per iteration: $O(U \cdot B \cdot A)$ — already computed for backpropagation; no extra forward pass
- Memory: 4 bytes (scalar accumulator)

**Interpretation**

- **Low** ($\to 0$): Network policy closely matches MCTS policy — the policy head has learned the search signal well.
- **High**: Network policy diverges from MCTS guidance — expected early in training or when the network cannot fit the target distribution.
- During training this value tends to decrease but rarely reaches zero because the network is a function approximator and cannot perfectly match the stochastic MCTS policy across all states.

**Typical Observations (this project)**

Empirical observations from training runs:
- Early iteration: 2.0–4.0 (network is random, far from MCTS targets)
- Mid training: 0.3–1.0 (network learning policy structure)
- Late training: 0.1–0.4 (policy convergence, but non-zero plateau remains)

**Notes**

- Already computed for backpropagation — zero additional cost.
- Uses $\ln$ (natural log), so values are in nats.
- The target $\pi_{\text{MCTS}}$ includes Dirichlet noise at the root during self-play, so the target is not the "pure" MCTS policy.
- Can spike if the replay buffer contains stale data from an older network.

---

### Value Loss

**Definition**

Mean squared error between the network's value prediction and the actual game outcome (backfilled self-play return $z$). This is the training signal for the value head.

**Formula**

$$L_{\text{value}} = \frac{1}{B}\sum_{i=1}^{B} \bigl(v_{\theta}(s_i) - z_i\bigr)^2$$

**Variables**

| Variable | Description |
|----------|-------------|
| $B$ | Batch size |
| $v_{\theta}(s_i)$ | Network's scalar value prediction for state $s_i$ |
| $z_i$ | Actual game outcome from self-play (backfilled returns: +1 = win, -1 = loss) |

**Implementation**

`src/Awale/Training.jl` — `train_step` function, line 196:
```julia
value_loss = Flux.mse(after_values, Y_v)
```
`after_values` contains the value head's predictions after the gradient update; `Y_v` contains the backfilled $z$ targets.

**Computational Cost**

- Per call: $O(B)$ — squared differences
- Per iteration: $O(U \cdot B)$ — free (reuses loss computation)
- Memory: 4 bytes (scalar accumulator)

**Interpretation**

- **Low** ($\to 0$): Value head predicts game outcomes accurately.
- **High**: Value head poorly calibrated — expected early in training when the network has not learned to evaluate positions.
- The value loss has a natural floor determined by the stochasticity of the game: the same position can lead to different outcomes depending on later play.

**Typical Observations (this project)**

Empirical observations from training runs:
- Early iteration: 0.5–2.0 (value head near random)
- Mid training: 0.2–0.8 (learning positional evaluation)
- Late training: 0.1–0.5 (some irreducible error remains due to game stochasticity)

**Notes**

- Already computed for backpropagation — zero additional cost.
- $z$ is the backfilled return from the completed game (outcome for the player to move), not the raw reward at the terminal state.
- Loss is computed AFTER the gradient update (on the updated model), not before.

---

### Gradient Norm

**Definition**

L2 norm of the gradient vector across all model parameters. Measures the magnitude of the gradient signal during backpropagation.

**Formula**

$$\|\nabla_{\theta} L\|_2 = \sqrt{\sum_{i} g_i^2}$$

**Variables**

| Variable | Description |
|----------|-------------|
| $g_i$ | $i$-th component of the gradient vector |

**Implementation**

`src/Awale/Training.jl` — `train_step` function, lines 199–221:
```julia
function flat_norm(x)
    if x isa Number
        return Float32(abs2(x))
    elseif x isa AbstractArray
        return Float32(sum(abs2, x))
    elseif x isa Tuple || x isa NamedTuple
        total = 0.0f0
        for field in x
            total += flat_norm(field)
        end
        return total
    ...
end
grad_norm = sqrt(flat_norm(grads))
```

**Computational Cost**

- Per call: $O(P)$ — recursively walks all gradient values
- Per iteration: $O(U \cdot P)$ — free (gradients already computed by backprop)
- Memory: 4 bytes (scalar accumulator)

**Interpretation**

- **Low**: Small gradients — may indicate saturation (value head saturated) or convergence (policy head settled).
- **High**: Large updates — may indicate instability, loss spikes, or the network encountering new, unexpected training data.
- The absolute scale depends on model size, loss magnitude, and batch size; trends matter more than absolute values.

**Typical Observations (this project)**

Empirical observations from training runs:
- Stable training: 0.1–5.0
- Spikes above 10.0 may occur during early iterations when the network is still random
- Sustained values > 15.0 warrant investigation

**Notes**

- Free computation — reuses the gradient from `Flux.gradient` in the training step.
- The `flat_norm` function recursively handles nested Tuple/NamedTuple structures from Flux's parameter representation.

---

### Parameter Update Norm

**Definition**

L2 norm of the change in the full parameter vector before and after a training iteration. Measures how much all model parameters (weights, biases, embeddings) changed as a result of the iteration's gradient updates.

**Formula**

$$\|\Delta\theta\|_2 = \|\theta_{\text{new}} - \theta_{\text{old}}\|_2$$

**Variables**

| Variable | Description |
|----------|-------------|
| $\theta_{\text{old}}$ | Flattened parameter vector before the training iteration |
| $\theta_{\text{new}}$ | Flattened parameter vector after the training iteration |

**Implementation**

`train.jl`, before and after `run_training_iteration` call (lines ~778–780):
```julia
new_params_vec, _ = Flux.destructure(model[])
param_update_norm = sqrt(sum((new_params_vec .- old_params_vec) .^ 2))
```
`old_params_vec` is captured via `Flux.destructure` before the iteration.

**Computational Cost**

- Per call: $O(P)$ — vector difference of all $P$ parameters
- Per iteration: $O(P)$ — one subtraction + one sum of squares
- Memory: $P \times 4$ bytes (shared with drift computation via `destructure` vector)

**Interpretation**

- **Low** ($\to 0$): Parameters barely changed — may indicate convergence, vanishing gradients, or very small learning rate relative to current location.
- **High** ($> 1.0$): Large parameter change — may indicate instability, a significant policy shift, or gradient spikes.
- Trends over iterations are more informative than absolute values.

**Typical Observations (this project)**

Empirical observations from training runs:
- Stable training: 0.001–0.05
- Active learning: 0.05–0.5
- Sustained values > 1.0 may indicate instability

**Notes**

- The `destructure` vector is also used for network drift — no additional capture call needed.
- Captures ALL parameters: weights, biases, embeddings; excludes BatchNorm parameters if present.
- This metric measures change over the full iteration (all gradient updates combined), not per-batch.

---

## 2. Policy Learning Metrics

How well the network's policy matches the MCTS teaching signal.

### KL Divergence

**Definition**

Kullback–Leibler divergence from the MCTS target policy to the network's predicted policy. Measures how much information the network has yet to learn from the search.

Direction: $D_{\text{KL}}(\pi_{\text{MCTS}} \parallel \pi_{\theta})$ — we use this direction because $\pi_{\text{MCTS}}$ is the teaching signal. The reverse direction $D_{\text{KL}}(\pi_{\theta} \parallel \pi_{\text{MCTS}})$ would be dominated by regions where the network assigns probability but MCTS does not, which is less informative for learning progress.

**Formula**

$$D_{\text{KL}}(\pi_{\text{MCTS}} \parallel \pi_{\theta}) = \sum_{a} \pi_{\text{MCTS}}(a|s) \cdot \ln\frac{\pi_{\text{MCTS}}(a|s)}{\pi_{\theta}(a|s)}$$

**Variables**

| Variable | Description |
|----------|-------------|
| $\pi_{\text{MCTS}}(a\|s)$ | Target policy from MCTS at state $s$ |
| $\pi_{\theta}(a\|s)$ | Network's predicted policy at state $s$ |
| $a$ | Action index (6 actions in Awale) |

**Implementation**

`src/Awale/Training.jl` — `run_training_iteration`, lines 348–350:
```julia
safe_target = clamp.(Y_pi, 1.0f-10, 1.0f0)
safe_pred = clamp.(after_probs, 1.0f-10, 1.0f0)
kl_per_pos = sum(Y_pi .* (log.(safe_target) .- log.(safe_pred)), dims = 1)
```
Mean, median, and percentiles (P25, P50, P75, P95) are computed from the per-position values at lines 422–425. Stored in `TrainingResult` as `kl_mean`, `kl_median`, `kl_p25`, `kl_p50`, `kl_p75`, `kl_p95`.

**Computational Cost**

- Per call: $O(B \cdot A)$ — element-wise operations on logits
- Per iteration: $O(N \cdot A)$ — computed per position during batch processing; no forward pass needed (reuses existing logits)
- Memory: ~$N \times 4$ bytes for per-position values (accumulated and summarized)

**Interpretation**

- **Low** ($\to 0$): Both policies are nearly identical — search may no longer be providing new information to learn.
- **High**: Network policy differs significantly from MCTS — search is still providing useful teaching signal.
- Very low KL combined with high Top-1 (> 95%) suggests the network has fully learned the MCTS policy.

**Typical Observations (this project)**

Empirical observations from training runs:
- Active learning mean KL: 0.05–0.50
- At ~60% Top-1: KL ≈ 0.12 (healthy search contribution)
- Late training mean KL: 0.02–0.10

**Notes**

- Values are in nats (natural log).
- Probabilities are clamped to $[10^{-10}, 1]$ to prevent $\ln(0)$ and numeric instability.
- KL divergence is non-negative and asymmetric; zero only when $\pi_{\text{MCTS}} = \pi_{\theta}$ everywhere.

---

### Top-K Agreement

**Definition**

Proportion of positions where the MCTS's top-1 action is contained in the network's top-$K$ actions. Measures whether the network ranks the search-preferred move highly enough.

**Important**: This checks set membership — "is the MCTS-preferred move in the network's top-$K$?" — NOT the intersection of two top-$K$ sets.

**Formula**

$$\text{Top-1} = \frac{1}{B}\sum_{i=1}^{B} \mathbb{1}\bigl[\arg\max \pi_{\text{MCTS}}(s_i) = \arg\max \pi_{\theta}(s_i)\bigr]$$

$$\text{Top-}K = \frac{1}{B}\sum_{i=1}^{B} \mathbb{1}\bigl[\arg\max \pi_{\text{MCTS}}(s_i) \in \text{top-}K(\pi_{\theta}(s_i))\bigr]$$

**Variables**

| Variable | Description |
|----------|-------------|
| $B$ | Total number of positions evaluated |
| $\arg\max \pi_{\text{MCTS}}(s_i)$ | Action with highest probability in MCTS policy at state $s_i$ |
| $\arg\max \pi_{\theta}(s_i)$ | Action with highest probability in network policy at state $s_i$ |
| $\text{top-}K(\pi_{\theta}(s_i))$ | Set of $K$ actions with highest network probability |

**Implementation**

`src/Awale/Training.jl` — `run_training_iteration`, lines 354–360:
```julia
target_order = sortperm(Y_pi, dims = 1, rev = true)
pred_order = sortperm(after_probs, dims = 1, rev = true)
target_top1 = vec(target_order[1, :])
for col = 1:size(Y_pi, 2)
    push!(all_top1, target_top1[col] == pred_order[1, col])
    push!(all_top2, target_top1[col] in pred_order[1:2, col])
    push!(all_top3, target_top1[col] in pred_order[1:3, col])
end
```
Stored in `TrainingResult` as `top1_pct`, `top2_pct`, `top3_pct`.

**Computational Cost**

- Per call: $O(B \cdot A \cdot \ln A)$ — sorting per position
- Per iteration: $O(N \cdot A \cdot \ln A)$
- Memory: ~$N \times 3$ bytes (one Bool per position per K-value, accumulated)

**Interpretation**

- **Low** (Top-1 < 40%): Network and MCTS frequently disagree on the best move — search is providing a clearly different signal than the network's prior.
- **High** (Top-1 > 95%): Network and MCTS nearly always agree — search may no longer improve the policy. At this point, KL is typically very low as well.
- Top-2 and Top-3 typically saturate faster than Top-1 and are less informative for convergence detection.

**Typical Observations (this project)**

Empirical observations from training runs:
- Active learning Top-1: 40–80%
- At KL ≈ 0.12: Top-1 ≈ 60% (healthy exchange between search and network)
- Late training Top-1: 80–95%

**Notes**

- This metric is NOT set intersection; it specifically tests whether the MCTS-preferred action is ranked highly by the network.
- The formula was previously documented as intersection of top-$K$ sets — the current implementation uses set membership (correct as of v2.0.0).
- Top-1 is the most informative signal; Top-2 and Top-3 are supplementary.

---

### Policy Distance (L1)

**Definition**

Mean absolute difference between the MCTS target policy and the network's predicted policy, averaged over all positions and actions. Scaled by $1/2$ so the range is $[0, 1]$ — a distance of 1 means policies are completely opposite.

**Formula**

$$L_1 = \frac{1}{2B}\sum_{i=1}^{B} \sum_{a} \bigl|\pi_{\text{MCTS}}(a|s_i) - \pi_{\theta}(a|s_i)\bigr|$$

**Variables**

| Variable | Description |
|----------|-------------|
| $B$ | Number of positions |
| $\pi_{\text{MCTS}}(a\|s_i)$ | MCTS target probability for action $a$ at state $s_i$ |
| $\pi_{\theta}(a\|s_i)$ | Network-predicted probability for action $a$ at state $s_i$ |

**Implementation**

`src/Awale/Training.jl` — `run_training_iteration`, lines 363–364:
```julia
l1_per_pos = sum(abs.(after_probs .- Y_pi), dims = 1) / 2
append!(all_l1_per_position, vec(l1_per_pos))
```
Mean and percentiles (P25, P50, P75, P95) computed at lines ~490. Stored in `TrainingResult` as `l1_mean`, `l1_p25`, `l1_p50`, `l1_p75`, `l1_p95`.

**Computational Cost**

- Per call: $O(B \cdot A)$ — element-wise absolute difference
- Per iteration: $O(N \cdot A)$ — free (reuses existing predictions)
- Memory: ~$N \times 4$ bytes (per-position values, accumulated)

**Interpretation**

- **Low** ($\to 0$): Policies are nearly identical — the network has learned the MCTS distribution.
- **High** ($\to 0.5$): Policies disagree substantially — search is providing a meaningfully different distribution than the network.
- L1 provides a complementary view to KL: KL is more sensitive to differences in high-probability actions; L1 gives equal weight to all actions.

**Typical Observations (this project)**

Empirical observations from training runs:
- Active learning: 0.05–0.35
- At KL ≈ 0.12: L1 ≈ 0.18 (consistent with healthy disagreement)
- Late training: 0.02–0.10

**Notes**

- The $1/2$ scaling factor ensures the metric is in $[0, 1]$ (a probability difference summed over 6 actions has a maximum of 2).
- L1 and KL are correlated but not deterministically linked: L1 measures raw probability mass difference, KL measures information divergence.

---

### Target Policy Entropy

**Definition**

Shannon entropy of the MCTS target policy. Measures how diffuse or concentrated the search policy is across actions.

**Formula**

$$H(\pi_{\text{MCTS}}) = -\sum_{a} \pi_{\text{MCTS}}(a|s) \cdot \ln \pi_{\text{MCTS}}(a|s)$$

**Variables**

| Variable | Description |
|----------|-------------|
| $\pi_{\text{MCTS}}(a\|s)$ | Target policy probability for action $a$ |

**Implementation**

`src/Awale/Training.jl` — `train_step` function, lines 227–228:
```julia
safe_target = clamp.(Y_pi, 1.0f-10, 1.0f0)
target_entropy = -sum(Y_pi .* log.(safe_target)) / size(X, 2)
```
Distributional stats (mean, min, max, P25, P50, P75, P95) computed in `run_training_iteration` from accumulated per-position values. Stored in `TrainingResult` as `entropy_mean`, `entropy_min`, `entropy_max`, `entropy_p25`, `entropy_p50`, `entropy_p75`, `entropy_p95`.

**Computational Cost**

- Per call: $O(A)$ — element-wise entropy
- Per iteration: $O(N \cdot A)$ — free (reuses target probabilities from loss)
- Memory: 4 bytes per position (accumulated)

**Interpretation**

- **Low** ($\to 0$): MCTS strongly favours a single move — the position likely has a clear best answer (near-deterministic policy).
- **High**: MCTS policy is diffuse — multiple moves are considered plausible. The theoretical maximum for 6 actions is $\ln 6 \approx 1.79$ nats (uniform distribution).
- Trends over time: entropy tends to decrease as the policy sharpens toward better moves.

**Typical Observations (this project)**

Empirical observations from training runs:
- Typical range: 0.0–1.79 (6-action maximum)
- Active learning mean: 0.5–1.2 nats
- Sharp positions (clear best move): < 0.3 nats

**Notes**

- Values are in nats (natural log). Maximum for $A$ actions is $\ln A$.
- Entropy is computed on the clamped target probabilities.
- Low entropy at root does not imply the whole tree is deterministic — it only reflects the root visit distribution.

---

### Predicted Policy Entropy

**Definition**

Shannon entropy of the network's predicted policy. Measures how confident or diffuse the network is about its action probabilities.

**Formula**

$$H(\pi_{\theta}) = -\sum_{a} \pi_{\theta}(a|s) \cdot \ln \pi_{\theta}(a|s)$$

**Variables**

| Variable | Description |
|----------|-------------|
| $\pi_{\theta}(a\|s)$ | Network-predicted probability for action $a$ |

**Implementation**

`src/Awale/Training.jl` — `train_step` function, line 224:
```julia
pred_entropy = -sum(after_probs .* after_log_probs) / size(X, 2)
```
Stored in `TrainingResult` as `avg_pred_entropy`.

**Computational Cost**

- Per call: $O(A)$ — element-wise entropy
- Per iteration: $O(N \cdot A)$ — free (reuses predicted logits from loss)
- Memory: 4 bytes (scalar average)

**Interpretation**

- **Low**: Network is confident in its action choice — one action dominates.
- **High**: Network is uncertain — multiple actions have similar probabilities.
- Compared to target entropy: predicted entropy is typically slightly higher (the network produces softer distributions than the sharper MCTS policy, especially early in training).

**Typical Observations (this project)**

Empirical observations from training runs:
- Typical range: 0.0–1.79 (same 6-action maximum as target)
- Active learning mean: 0.6–1.4 nats (usually 0.1–0.3 nats higher than target entropy)
- Late training: 0.3–0.8 nats (network becomes more confident)

**Notes**

- Values are in nats (natural log).
- The difference $H(\pi_{\theta}) - H(\pi_{\text{MCTS}})$ is a useful diagnostic: if it grows, the network is becoming relatively more uncertain than the search suggests is warranted.
- The network entropy cannot exceed the maximum possible entropy for $A=6$ actions ($\ln 6 \approx 1.79$ nats).

---

## 3. Search Metrics

Metrics from the MCTS search process itself — how decisive, calibrated, and impactful the search is.

### Root Confidence

**Definition**

Ratio of the most-visited action's visit count to total visits at the MCTS root node. Measures how decisive the search is about the best move.

**Formula**

$$\text{RootConf} = \frac{\max_a N(a)}{\sum_a N(a)}$$

**Variables**

| Variable | Description |
|----------|-------------|
| $N(a)$ | Visit count for action $a$ at the root |
| $a$ | Action index (1–6) |

**Implementation**

`src/Awale/MCTS.jl` — `search_with_stats` function, line 186:
```julia
return best_action, counts, maximum(counts), root_q, root_value
```
Where `counts` is $N(a) / \sum N(a)$. The third return value (`maximum(counts)`) is `max_a N(a) / sum_a N(a)`. Accumulated in `all_root_confidences` in `Training.jl`. Distributional stats (mean, min, max, P25, P50, P75, P95) stored in `TrainingResult`.

**Computational Cost**

- Per call: $O(1)$ — simple ratio of visit counts
- Per iteration: $O(N)$ — one ratio per position
- Memory: 4 bytes per position (accumulated)

**Interpretation**

- **Low** ($\to 0.2$): Multiple moves appear equally promising after search — the position is highly uncertain.
- **High** ($\to 1.0$): MCTS strongly favours one move — the position likely has a clear best answer.
- The theoretical minimum for 6 actions (uniform visits) is $\frac{1}{6} \approx 0.17$.

**Typical Observations (this project)**

Empirical observations from training runs:
- Typical mean: 0.4–0.7
- ~0.55 mean is common for Awale, where multiple competitive moves exist in most positions
- Full range observed: 0.2–1.0

**Notes**

- Root confidence depends on the number of simulations: more sims tend to increase confidence (the search converges on the best move).
- This metric measures decisiveness, not correctness — a confident wrong move is still wrong.

---

### Root Q (after search)

**Definition**

The value at the root node after MCTS search completes, computed as the mean value across all visited children (weighted by visit count). This is the search-improved value estimate, incorporating lookahead from all simulations.

**Formula**

$$Q_{\text{root}} = \frac{\sum_{n \in \text{children}} v_n \cdot N_n}{V_{\text{root}}}$$

**Variables**

| Variable | Description |
|----------|-------------|
| $v_n$ | Value estimate of child node $n$ |
| $N_n$ | Visit count of child node $n$ |
| $V_{\text{root}}$ | Total visits at root |

**Implementation**

`src/Awale/MCTS.jl` — `search_with_stats` function, line 186:
```julia
root_q = root.value_sum[] / max(1, root.visits[])
return best_action, counts, maximum(counts), root_q, root_value
```
Accumulated across all self-play positions in `all_root_q` in `Training.jl`. Stored in `TrainingResult` as `root_q_mean`.

**Computational Cost**

- Per call: $O(1)$ — simple division of already-accumulated values
- Per iteration: $O(N)$ — one value per position
- Memory: 4 bytes per position (accumulated)

**Interpretation**

- $Q_{\text{root}} \in [-1, 1]$ — the range of game outcomes in Awale.
- **Closer to +1**: Search predicts a strong advantage for the player to move.
- **Closer to -1**: Search predicts a strong disadvantage.
- This value incorporates the lookahead from MCTS — it is NOT the raw network value (see Search Gain for the comparison).

**Typical Observations (this project)**

Empirical observations from training runs:
- Typical mean: varies with match strength; often near 0.0 (balanced positions)
- At promotion level: may skew positive as the model plays better than random
- Full range: $[-1, 1]$

**Notes**

- There are TWO relevant root values:
  1. **Raw network value** $V_{\text{network}}$: the value head's prediction before any MCTS simulation (captured at `MCTS.jl:143`).
  2. **Root Q after search** $Q_{\text{root}}$: the MCTS value at root after all simulations.
- $Q_{\text{root}}$ is always the search-improved estimate; the gap $Q_{\text{root}} - V_{\text{network}}$ measures search improvement (see Search Gain).

---

### Search Gain

**Definition**

The improvement MCTS search provides over the raw network value estimate. Positive values mean search finds a better evaluation than the network alone; values near zero mean the network's prior is already accurate.

**Formula**

$$\text{SearchGain} = Q_{\text{root}} - V_{\text{network}}$$

Where $Q_{\text{root}}$ is the MCTS value at root after search and $V_{\text{network}}$ is the value head's raw prediction before any search.

**Variables**

| Variable | Description |
|----------|-------------|
| $Q_{\text{root}}$ | Search-improved value estimate at root after MCTS completes |
| $V_{\text{network}}$ | Raw network value prediction before MCTS (first forward pass) |

**Implementation**

`train.jl`, after `run_training_iteration` (line ~789 area):
```julia
search_gain = root_q_mean - network_value_mean
```
Where `root_q_mean` and `network_value_mean` come from `TrainingResult`. The raw value is captured at `src/Awale/MCTS.jl:143`:
```julia
logits, root_value = predict_inference(mcts.model, root.state)
```

**Computational Cost**

- Per call: $O(1)$ — scalar difference
- Per iteration: $O(1)$ — one subtraction
- Memory: 4 bytes (scalar)
- Zero extra inference: the raw value is already computed during the first MCTS expansion

**Interpretation**

- **Positive** ($> 0$): Search improves over the raw network evaluation — MCTS finds a better value by lookahead.
- **Zero** ($\approx 0$): Search and network agree — the network's prior is already accurate.
- **Negative** ($< 0$): Search apparently degrades the value estimate (unusual in well-tuned search; may indicate insufficient simulations or a pathological tree).
- **Growing** over time: Search contribution is increasing relative to the network (possible if the value head is not keeping pace with policy improvement).

**Typical Observations (this project)**

Empirical observations from training runs:
- Typical mean during active learning: 0.02–0.15
- Late training: 0.0–0.05 (network converges toward search)

**Notes**

- This metric was added in v2.0.0 (see CR-O1). Previously only $Q_{\text{root}}$ was available for comparison.
- Both $Q_{\text{root}}$ and $V_{\text{network}}$ are in $[-1, 1]$, so Search Gain is in $[-2, 2]$ but typically in $[0, 0.3]$.
- Zero extra compute: both values are already produced by the existing MCTS and training pipeline.

---

## 4. Value Metrics

Value head calibration and accuracy.

### Value Loss

Defined and documented in **Optimization Metrics** (Section 1). The value loss is the mean squared error between the value head's predictions and the actual game outcomes.

See: [Value Loss](#value-loss) in Section 1.

### Value Calibration

**Definition**

Set of three metrics that measure how well the value head's predictions correlate with actual game outcomes:

1. **MAE** (Mean Absolute Error): average absolute prediction error.
2. **Pearson $r$**: linear correlation between predictions and targets.
3. **Spearman $\rho$**: rank correlation between predictions and targets (monotonic relationship, not requiring linearity).

Values are accumulated across all batches in the iteration (not just the last batch).

**Formulas**

$$\text{MAE} = \frac{1}{B}\sum_{i=1}^{B} |v_{\theta}(s_i) - z_i|$$

$$r = \frac{\sum_{i=1}^{B} (v_i - \bar{v})(z_i - \bar{z})}{\sqrt{\sum_{i=1}^{B} (v_i - \bar{v})^2 \sum_{i=1}^{B} (z_i - \bar{z})^2}}$$

$$\rho = r(\text{rank}(v), \text{rank}(z))$$

**Variables**

| Variable | Description |
|----------|-------------|
| $v_{\theta}(s_i)$ | Network value prediction for state $s_i$ |
| $z_i$ | Actual game outcome (backfilled return) |
| $\text{rank}(v)$ | Rank-transformed predictions |

**Implementation**

`src/Awale/Metrics.jl` — `compute_value_calibration` function, lines 264–280:
```julia
function compute_value_calibration(v_pred::AbstractVector{Float32}, v_target::AbstractVector{Float32})
    if std(v_pred) == 0.0f0 || std(v_target) == 0.0f0
        return (mae=NaN, pearson_r=NaN, spearman_rho=NaN)
    end
    mae = mean(abs.(v_pred .- v_target))
    pearson_r = cor(v_pred, v_target)
    pred_ranks[sortperm(v_pred)] .= Float64.(1:n)
    target_ranks[sortperm(v_target)] .= Float64.(1:n)
    spearman_rho = cor(pred_ranks, target_ranks)
    return (mae=mae, pearson_r=pearson_r, spearman_rho=spearman_rho)
end
```
Called from `train.jl` line ~798. The predictions and targets are accumulated via `append!` across all batches in the iteration (see `src/Awale/Training.jl` lines 340–341).

**Computational Cost**

- Per call: $O(B \ln B)$ for ranking (Spearman), $O(B)$ for MAE and Pearson
- Per iteration: $O(B_{\text{total}} \ln B_{\text{total}})$ where $B_{\text{total}}$ is all samples across all batches
- Memory: $O(B_{\text{total}})$ — stores all predictions and targets from the iteration

**Interpretation**

- **MAE low** ($< 0.3$): Value head predicts outcomes accurately on average.
- **MAE high** ($> 0.5$): Value head is poorly calibrated — predictions are on average off by more than 0.5 on the $[-1, 1]$ scale.
- **Pearson $r$ positive** ($> 0.3$): Predictions correlate linearly with outcomes.
- **Pearson $r$ near 0 or negative**: Value head is not correlated with outcomes — essentially random.
- **Spearman $\rho$ positive** ($> 0.3$): Predictions rank-order positions correctly (monotonic relationship).
- $|r|$ is typically slightly higher than $|\rho|$ when the relationship is linear; $\rho$ is more robust to outliers.

**Typical Observations (this project)**

Empirical observations from training runs:
- MAE during active training: 0.3–0.8
- MAE at late training: 0.2–0.5
- Pearson $r$ during active training: 0.3–0.7
- Spearman $\rho$ during active training: 0.3–0.7

**Notes**

- Returns `(NaN, NaN, NaN)` when either input has zero standard deviation (degenerate case: all predictions or targets are identical).
- Accumulated across ALL batches in the iteration (not just the last batch) as of v2.0.0 (CR-M1 fix). Previously only the last batch was used.
- No extra forward passes — uses the value predictions already computed during training.

---

## 5. Replay Metrics

Data pipeline health.

### Replay Fill %

**Definition**

Percentage of the circular replay buffer currently occupied. Measures how much training data is available for sampling.

**Formula**

$$\text{Replay Fill \%} = \frac{|\text{ReplayBuffer}|}{\text{Capacity}} \times 100$$

**Variables**

| Variable | Description |
|----------|-------------|
| $\|\text{ReplayBuffer}\|$ | Number of experiences currently stored in the buffer |
| $\text{Capacity}$ | Maximum capacity of the buffer |

**Implementation**

`src/Awale/Training.jl` — `run_training_iteration`, line 396:
```julia
replay_pct = round(replay_fill / replay_capacity * 100, digits=1)
```
Stored in `TrainingResult` as `replay_pct`.

**Computational Cost**

- Per call: $O(1)$ — simple ratio
- Per iteration: $O(1)$
- Memory: 8 bytes (Float64)

**Interpretation**

- **0%**: Buffer is empty — no training data available (first iteration).
- **100%**: Buffer is full — the oldest experiences are being overwritten by new ones.
- During early iterations, the fill percentage grows monotonically from 0 to 100%.
- After reaching 100%, the buffer is a fixed-size sliding window of the most recent experiences.

**Typical Observations (this project)**

Empirical observations from training runs:
- Iteration 1: < 5% (buffer just starting to fill)
- Reaches 100% after enough self-play games to fill the buffer
- Remains at 100% thereafter

**Notes**

- This is NOT the fraction of the state space explored, nor the fraction of samples consumed. It is simply how full the circular buffer is.
- The metric was formerly named "Replay Coverage" (renamed because it measures buffer fill, not state-space coverage).
- During bootstrap (fill < 10%), some dashboard states report `BOOTSTRAP` to indicate insufficient data for representative sampling.

---

## 6. Promotion Metrics

Model improvement milestones — how often and by how much the model improves.

### Win Rate

**Definition**

The win rate of the candidate model (current training model) against the best model (best-performing model so far) in arena matches. Used to determine whether the candidate model is stronger and should be promoted.

**Formula**

$$\text{WR}_{\text{candidate}} = \frac{W + 0.5D}{W + L + D}$$

**Variables**

| Variable | Description |
|----------|-------------|
| $W$ | Wins by the candidate model |
| $L$ | Losses by the candidate model |
| $D$ | Draws between candidate and best model |

**Implementation**

`train.jl` — arena match loop (candidate vs best). The win rate is computed in the promotion gate logic roughly every `ARENA_EVAL_ITER` iterations. Used as input to `update_elo!` and `PromotionGate`.

**Computational Cost**

- Per evaluation: $O(G_{\text{arena}})$ where $G_{\text{arena}}$ is the number of arena games
- Per iteration: not every iteration (configurable frequency)
- Memory: trivial (few scalars)

**Interpretation**

- **WR > 0.55**: Candidate is likely stronger than the best model (depending on number of games and statistical significance).
- **WR ≈ 0.50**: Candidate and best are equally matched.
- **WR < 0.50**: Candidate is weaker — expected most of the time as the best model was previously promoted.

**Typical Observations (this project)**

Empirical observations from training runs:
- Most iterations: 0.30–0.55 (candidate is usually weaker or equal)
- Promotion events: > 0.55 (candidate clearly outperforms best)

**Notes**

- Not every iteration evaluates win rate — arena matches are run at a configurable interval.
- The number of arena games per evaluation determines statistical significance.

---

### Elo Rating

**Definition**

Standard Elo rating system tracking the candidate and best model strengths. Updated after each arena evaluation using the match results.

**Formula**

$$E_{\text{candidate}} = \frac{1}{1 + 10^{(\text{rating}_{\text{best}} - \text{rating}_{\text{candidate}}) / 400}}$$

$$S_{\text{candidate}} = \frac{W + 0.5D}{W + L + D}$$

$$\Delta = K \cdot (S - E)$$

**Variables**

| Variable | Description |
|----------|-------------|
| $\text{rating}_{\text{best}}$ | Elo rating of the best (promoted) model |
| $\text{rating}_{\text{candidate}}$ | Elo rating of the current candidate model |
| $K$ | K-factor (update sensitivity); default: 64 |
| $W, L, D$ | Wins, losses, draws of candidate vs best |

**Implementation**

`src/Awale/Metrics.jl` — `EloTracker` struct and `update_elo!` function, lines 54–61:
```julia
function update_elo!(tracker::EloTracker, wins::Int, losses::Int, draws::Int, iteration::Int)
    total = wins + losses + draws
    total == 0 && return
    score = (wins + 0.5 * draws) / total
    expected = 1.0 / (1.0 + 10.0 ^ ((tracker.best_rating - tracker.candidate_rating) / 400.0))
    delta = tracker.k * (score - expected)
    tracker.candidate_rating += delta
end
```
`K = 64` (default). Logged in JSONL as `elo_candidate`, `elo_best`.

**Computational Cost**

- Per iteration: $O(1)$ — simple arithmetic
- Memory: 24 bytes (two Float64 ratings + one Float64 K-factor + history)

**Interpretation**

- **Candidate rating rising** ($\Delta > 0$): Candidate is improving relative to the best model.
- **Candidate rating falling** ($\Delta < 0$): Candidate is underperforming — training might be regressing or the current best is very strong.
- **Large Elo gap** (candidate $\ll$ best): Expected early in training before the first promotion.
- On promotion, `promote_elo!` sets `best_rating = candidate_rating`, resetting the gap to zero.

**Typical Observations (this project)**

Empirical observations from training runs:
- Starting ratings: 1500 (both candidate and best)
- Typical delta per evaluation: ±1–10
- Upset magnitude $|S - E|$ measures how surprising the match result was

**Notes**

- Elo uses $\log_{10}$ (base 10), not natural log — this is standard for Elo systems.
- The K-factor was increased from 32 to 64 for faster diagnostic signal (the gap closes more quickly after promotions).
- Elo values are meaningful for relative comparison within the same training run but not comparable across runs with different K-factors or arena configurations.

---

### Promotion History

**Definition**

Record of every promotion event: when the candidate model was promoted to become the new best model. Tracks how frequently the model improves.

**Implementation**

`src/Awale/Metrics.jl` — `PromotionRecord` struct and `print_promotion_event` function. Promotions are logged to `promotion_history.toml` for resume support. The `ProgressTracker` tracks `total_promotions` and `last_best_iter`.

Printed in the health dashboard as:
```
Promo: <iters_since_promotion>/<total_promotions>
```

**Computational Cost**

- Per promotion: $O(1)$
- Memory: negligible

**Interpretation**

- **Frequent promotions** (every 10–30 iterations): Model is improving rapidly — early training or a breakthrough.
- **Infrequent promotions** (50+ iterations between): Model improvement is slowing — may be approaching convergence.
- **No promotions** for a long period: Model may have stalled or converged.

**Typical Observations (this project)**

Empirical observations from training runs:
- First promotion: within 5–20 iterations (rapid early improvement)
- Subsequent promotions: every 10–50 iterations (slowing over time)

**Notes**

- The promotion gate uses a combination of win rate threshold and Elo comparison — see PromotionGate logic in `train.jl`.
- A promotion resets the Elo gap (best rating = candidate rating) for the next evaluation cycle.

---

## 7. Network Evolution Metrics

How the model changes over time — drift, stability, and diagnostic health indicators.

### Network Drift

**Definition**

KL divergence between the network's policy before and after a training iteration, measured over a **fixed reference set** $\mathcal{R}$ of 200 game states generated at startup with a fixed seed. The reference set is fixed so drift measurements are comparable across iterations.

**Formula**

$$D_{\text{drift}} = \frac{1}{|\mathcal{R}|}\sum_{s \in \mathcal{R}} D_{\text{KL}}\bigl(\pi_{\theta_{\text{before}}}(s) \parallel \pi_{\theta_{\text{after}}}(s)\bigr)$$

**Variables**

| Variable | Description |
|----------|-------------|
| $\mathcal{R}$ | Fixed reference set of 200 game states |
| $\pi_{\theta_{\text{before}}}(s)$ | Network policy at state $s$ before the training iteration |
| $\pi_{\theta_{\text{after}}}(s)$ | Network policy at state $s$ after the training iteration |

**Implementation**

`train.jl`, lines 760–792:
```julia
# Before iteration
before_logits, _ = with_inference_mode(() -> predict_raw(model[], drift_X), model[])

# After iteration
after_logits, _ = with_inference_mode(() -> predict_raw(model[], drift_X), model[])
before_log_probs = Flux.logsoftmax(before_logits, dims=1)
after_probs_drift = softmax(after_logits, dims=1)
drift_kl = sum(exp.(before_log_probs) .* (before_log_probs .- log.(clamp.(after_probs_drift, 1.0f-10, 1.0f0))), dims=1)
avg_drift_kl = sum(drift_kl) / length(drift_kl)
```

The reference set $\mathcal{R}$ is generated at startup (line 688):
```julia
reference_states = GameState[]
# ... collection from random play ...
if length(reference_states) > 200
    reference_states = reference_states[1:200]
end
```

**Computational Cost**

- Per iteration: $O(R \cdot A)$ — one forward pass on $R = 200$ states
- Memory: $R \times A \times 4$ bytes (logits for 200 states × 6 actions)
- One extra forward pass per iteration beyond the training loop

**Interpretation**

- **Low** ($\to 0$): Model is not changing much between iterations — may indicate convergence or very small updates.
- **High** ($> 0.05$): Policy is changing rapidly — possible instability or large parameter updates.
- Drift measures policy change, not improvement — a model can drift without improving.

**Typical Observations (this project)**

Empirical observations from training runs:
- Stable training: 0.001–0.02
- Observed typical: 0.003–0.007 (very stable in this project)
- Possible instability: > 0.05

**Notes**

- The direction is $D_{\text{KL}}(\pi_{\text{before}} \parallel \pi_{\text{after}})$ — divergence from before to after.
- The reference set is generated with a fixed seed (`BOOTSTRAP_RNG_SEED`), ensuring reproducibility across runs.
- Probabilities are clamped to $[10^{-10}, 1]$ to prevent $\ln(0)$.
- Drift == 0.0 exactly would mean zero parameter change — practically never occurs.
- Drift < $10^{-6}$ triggers a "network drift near zero" warning.

---

### Convergence Detection

**Definition**

Sliding window analysis of key metrics to determine whether training signals have stabilized. Each signal is classified as ACTIVE (still changing), STALLED (stable), or BOOTSTRAP (insufficient data).

**Detection Logic**

| Signal | Threshold $\varepsilon$ | State | Condition |
|--------|------------------------|-------|-----------|
| KL variance | 0.001 | ACTIVE | $\text{var}(\text{KL}_{[i-W:i]}) > \varepsilon$ |
| | | STALLED | $\text{var}(\text{KL}_{[i-W:i]}) \leq \varepsilon$ |
| Drift variance | 0.0001 | ACTIVE | $\text{var}(\text{drift}_{[i-W:i]}) > \varepsilon$ |
| | | STALLED | $\text{var}(\text{drift}_{[i-W:i]}) \leq \varepsilon$ |
| Top1 variance | 1.0 (%) | ACTIVE | $\text{var}(\text{Top1}_{[i-W:i]}) > \varepsilon$ |
| | | STALLED | $\text{var}(\text{Top1}_{[i-W:i]}) \leq \varepsilon$ |
| Param update norm variance | $10^{-8}$ | ACTIVE | $\text{var}(\text{param\_update}_{[i-W:i]}) > \varepsilon$ |
| | | STALLED | $\text{var}(\text{param\_update}_{[i-W:i]}) \leq \varepsilon$ |

Where $W = \min(20, \text{available\_iters})$. All signals report `BOOTSTRAP` when fewer than 5 iterations are available.

**Implementation**

`train.jl`, lines 939–954:
```julia
window_size = min(20, length(ma_buf_kl))
if window_size >= 5
    kl_window = ma_buf_kl[end-window_size+1:end]
    drift_window = ma_buf_drift[end-window_size+1:end]
    top1_window = ma_buf_top1[end-window_size+1:end]
    param_window = ma_buf_param_update[end-window_size+1:end]
    kl_stable = var(kl_window) < 0.001f0 ? "STALLED" : "ACTIVE"
    drift_stable = var(drift_window) < 0.0001f0 ? "STALLED" : "ACTIVE"
    top1_stable = var(top1_window) < 1.0f0 ? "STALLED" : "ACTIVE"
    param_stable = var(param_window) < 1.0f-8 ? "STALLED" : "ACTIVE"
else
    kl_stable = "BOOTSTRAP"; drift_stable = "BOOTSTRAP"
    top1_stable = "BOOTSTRAP"; param_stable = "BOOTSTRAP"
end
```

**Computational Cost**

- Per iteration: $O(W)$ — compute variance over the sliding window
- Memory: $W \times 4$ bytes per signal buffer

**Interpretation**

- **ALL ACTIVE**: Training is changing meaningfully — all metrics are varying above noise floor.
- **ALL STALLED**: Every signal has flattened — strong evidence the model has converged.
- **Mixed**: Some signals stable, others active — partial convergence. For example, KL and drift stable but param updates still active suggests the policy is settled but weights are still adjusting.
- **BOOTSTRAP**: First 5 iterations — insufficient data for meaningful variance estimates.

**Typical Observations (this project)**

Empirical observations from training runs:
- Early training: all signals ACTIVE
- Mid training: mixed (some signals stabilize)
- Late training: increasing number of STALLED signals

**Notes**

- Passive indicators only — no circuit breakers or behavioral changes.
- The parameter update window uses a dedicated buffer (`ma_buf_param_update`) containing actual `param_update_norm` values, not a proxy signal (as of v2.0.0 — see O6 fix).

---

### Training Health Dashboard

**Definition**

Single-line summary per iteration that provides an at-a-glance assessment of training health across five axes: network learning, search usefulness, model drift, promotion progress, replay fill, and value calibration.

**Format**

```
Net:<state> Srch:<state> Drift:<state> Promo:<since>/<total> Rply:<fill>% ValCal:<state>
```

**Example**

```
Net:ACTIVE Srch:HIGH Drift:LOW Promo:3/5 Rply:84.8% ValCal:OK
```

**Implementation**

`train.jl`, lines 957–978. See Health Dashboard Reference below for all states, triggers, and threshold rationale.

**Computational Cost**

- Per iteration: $O(1)$ — simple comparisons
- Memory: 0 (transient)

---

### Diagnostic Warnings

**Definition**

Passive warnings printed when certain thresholds are breached. All warnings are informational — no circuit breakers or behavioral changes.

**Warning Rules**

| Trigger | Message |
|---------|---------|
| Top-1 > 95% | `⚠ Top-1 agreement N% — search may no longer improve policy` |
| Drift KL < $10^{-6}$ | `⚠ Network drift near zero — training may have converged` |
| Parameter update norm > 5× rolling mean loss | `⚠ Parameter update unusually large — possible instability` |

**Implementation**

`train.jl`, lines 981–998:
```julia
warning_messages = String[]
if training_result.top1_pct > 95.0f0
    msg = "Top-1 agreement $(round(training_result.top1_pct, digits=1))% — search may no longer improve policy"
    push!(warning_messages, msg)
end
if avg_drift_kl < 1.0f-6
    msg = "Network drift near zero — training may have converged"
    push!(warning_messages, msg)
end
rolling_mean = ma_win(ma_buf_policy_loss, 5)
if isfinite(rolling_mean) && rolling_mean > 0.0f0 && param_update_norm > 5.0f0 * rolling_mean
    msg = "Parameter update unusually large — possible instability"
    push!(warning_messages, msg)
end
```

**Computational Cost**

- Per iteration: $O(1)$ — three comparisons
- Memory: transient (strings allocated and printed)

**Interpretation**

- **Top-1 warning**: Fires when the network and MCTS nearly always agree — the search signal is saturated and may no longer be improving the policy.
- **Drift warning**: Fires when the model barely changes between iterations — may indicate convergence or a dead training loop.
- **Param update warning**: Fires when a single iteration produces an unusually large parameter change — possible instability, gradient explosion, or data distribution shift.

**Notes**

- All warnings are purely informational — no circuit breakers or behavioral changes.
- Warnings are accumulated into a `warning_messages` array and written to JSONL as a combined string.

---

## Interpreting Metric Combinations

Individual metrics are informative, but the combination of signals across categories provides a richer picture of training dynamics.

### Case 1: Network converging to MCTS policy

**Signal pattern**

| Metric | Direction |
|--------|-----------|
| KL Divergence | ↓ Decreasing |
| Top-1 Agreement | ↑ Increasing |
| Policy Distance (L1) | ↓ Decreasing |
| Network Drift | ↓ Decreasing |

**Interpretation**

The network is progressively approximating the MCTS policy. Search is providing less new information per iteration because the network has already learned what the search knows.

**What this means**

- Healthy convergence — the training loop is working as designed.
- If KL → 0 and Top-1 → 100%, the network has essentially "caught up" to the search, and the search may no longer be improving the policy.
- At this point, additional search simulations may yield diminishing returns.

### Case 2: Active search contribution

**Signal pattern**

| Metric | State |
|--------|-------|
| KL | Moderate–high (0.1–0.5) |
| Top-1 | Moderate (40–80%) |
| Root Confidence | Moderate–high (0.4–0.7) |
| Search Gain | Positive (0.02–0.15) |

**Interpretation**

MCTS consistently finds policies that differ significantly from the network's prior. The search is actively contributing to the training signal.

**What this means**

- This is the desired regime during active learning.
- The network is learning from a search that genuinely improves over its prior.
- If KL stays high for many iterations without decreasing, the network may have hit a capacity limit (see Phase 1–4 decision protocol in the project's experimental protocol).

### Case 3: Rapid training changes

**Signal pattern**

| Metric | Direction |
|--------|-----------|
| Gradient Norm | ↑ Increasing or spiking |
| Parameter Update Norm | ↑ Elevated |
| Network Drift | ↑ Elevated (> 0.05) |
| Convergence signals | Mixed (some STALLED, some ACTIVE) |

**Interpretation**

Training is changing rapidly — the model parameters and policy are shifting significantly between iterations.

**What this means**

- May indicate normal learning at a critical point (e.g., discovering a new strategy).
- May indicate instability — especially if paired with loss spikes, value calibration degradation, or Elo drops.
- Cross-check with value calibration: if MAE is rising while drift is high, the value head may not be keeping up with policy changes.
- High drift sustained over many iterations (> 10) warrants investigation.

### Case 4: Possible convergence

**Signal pattern**

| Metric | State |
|--------|-------|
| Network Drift | Very low (< 0.001) |
| Parameter Update Norm | Very low (< 0.001) |
| KL | Stable (low variance) |
| Top-1 | Stable (low variance) |
| Convergence signals | Mostly STALLED |

**Interpretation**

The model is changing very slowly across all metrics. Most training signals have flattened.

**What this means**

- Does NOT mean training is done — the model could be stuck in a local optimum.
- Does NOT mean the model won't improve — learning can resume if the replay buffer cycles to different data or if hyperparameters change.
- A plateau is a signal to evaluate: try more self-play data, adjust learning rate, or consider architectural changes (per the project's Phase 1–4 experimental protocol).
- Cross-check with Elo and Win Rate: if the model is still winning promotions despite low drift, it may genuinely have converged to a strong policy.

---

## Health Dashboard Reference

### Dashboard State Table

| Field | States | Triggers |
|-------|--------|----------|
| **Net** | ACTIVE / STALLED | $\Delta\text{KL} > 0.01$ OR $\|\Delta\theta\|_2 > 0.01$ → ACTIVE; else STALLED |
| **Srch** | HIGH / LOW | $\text{Top1} \leq 60\%$ AND $\text{KL} \leq 0.15$ AND $\text{L1} \leq 0.20$ → LOW; else HIGH |
| **Drift** | LOW / MEDIUM / HIGH | $< 0.01$ → LOW; $< 0.05$ → MEDIUM; $\geq 0.05$ → HIGH |
| **ValCal** | OK / HIGH / N/A | MAE $< 0.5$ → OK; MAE $\geq 0.5$ → HIGH; NaN → N/A |
| **Convergence** | ACTIVE / STALLED / BOOTSTRAP | Per-signal variance checks (see Convergence Detection); BOOTSTRAP when window $< 5$ |

### Threshold Rationale

Each threshold in the dashboard is chosen based on empirical observation and practical significance:

#### Net: $\Delta\text{KL} > 0.01$ OR $\|\Delta\theta\|_2 > 0.01$

- **KL 0.01 threshold**: This is approximately the KL convergence floor observed during stable training. Below 0.01, the policy change between iterations is negligible — the model is essentially at a fixed point. Above 0.01, the network is actively learning from the iteration's gradient updates.
- **Param update norm 0.01 threshold**: Corresponds to a parameter change of about 1% of the typical parameter magnitude. Below this threshold, the gradient updates are not meaningfully shifting the weights. The OR condition ensures Net is ACTIVE if either signal shows meaningful change.

#### Srch: Top1 $\leq$ 60% AND KL $\leq$ 0.15 AND L1 $\leq$ 0.20

- **Top1 60% (originally 80%)**: The original single-signal check used Top1 < 80% → HIGH (search providing signal). This was adjusted to a triple conjunction because a single signal can trigger false HIGH. For example, Top1 < 80% alone is normal in early training and doesn't necessarily mean search is useful — the network just hasn't learned yet. The 60% threshold is the empirically observed midpoint where search is clearly providing different rankings than the network.
- **KL 0.15 threshold**: At KL > 0.15, the target and predicted policies are substantively different. Below 0.15, the policies are similar enough that search is primarily confirming the network's prior rather than providing new information.
- **L1 0.20 threshold**: Corresponds to approximately 20% average probability mass difference per action. Below this, the distributional shapes are similar enough that search benefit is limited.
- **Triple conjunction logic**: The dashboard reports Srch:LOW (search usefulness is low) only when ALL three signals are below their thresholds. If ANY signal exceeds its threshold, search is providing genuinely new information → Srch:HIGH. This prevents false LOW readings when only one signal type shows agreement.

#### Drift: $< 0.01$ LOW, $< 0.05$ MEDIUM, $\geq 0.05$ HIGH

- **0.01 threshold (< 0.01 → LOW)**: Below 0.01 KL drift, the policy change per iteration is below the level of meaningful policy transformation. This is typical of stable, incremental learning.
- **0.05 threshold (< 0.05 → MEDIUM, $\geq 0.05$ → HIGH)**: A drift of 0.05 KL means the policy has shifted enough that action probabilities change by roughly 5–10 percentage points on average. Below 0.05, drift is within the range of normal learning. Above 0.05, the model is changing rapidly and warrants attention — the policy may be in a phase of rapid discovery or instability.

#### ValCal: MAE $< 0.5$ → OK

- **MAE 0.5 threshold**: On the $[-1, 1]$ outcome scale, an MAE of 0.5 corresponds to roughly 25% of the full range. If the value head is off by more than 0.5 on average, it is poorly calibrated — predictions are not reliably distinguishing winning from losing positions. Below 0.5, the value head has meaningful predictive signal.
- Returns N/A when MAE is NaN (degenerate case: all predictions or targets identical).

#### Variance thresholds (Convergence Detection)

| Signal | Threshold | Rationale |
|--------|-----------|-----------|
| KL variance | 0.001 | Typical KL values are in [0.01, 0.5]. Variance below 0.001 means KL is essentially flat — the policy is not changing its divergence from MCTS. |
| Drift variance | 0.0001 | Drift values are typically in [0.001, 0.02]. Variance below 0.0001 means drift is flat — the per-iteration policy change is constant. |
| Top1 variance | 1.0 (%) | Top1 in percent. Variance below 1% means the agreement rate is essentially stable. |
| Param update norm variance | $10^{-8}$ | Parameter update norm is typically $O(10^{-3})$ to $O(10^{-1})$. Variance below $10^{-8}$ means the update magnitude is essentially constant. |

These thresholds are set to detect the noise floor of each metric during stable training. When variance drops below the threshold, the metric is no longer providing meaningful variation — it has "flatlined."

#### BOOTSTRAP (window $<$ 5)

- **Window $<$ 5**: With fewer than 5 data points, variance estimates are too noisy to be reliable. The dashboard reports BOOTSTRAP for all convergence signals until the 6th iteration.
- **Replay fill $<$ 10%**: The `Net` state may also effectively be BOOTSTRAP if the replay buffer is too small for representative sampling, though the current implementation does not use a hard BOOTSTRAP state for Net (it reports ACTIVE/STALLED based on available signals).

### Convergence Stability States

```
KL:<state> Drift:<state> Top1:<state> Param:<state>
```

Each convergence signal reports one of three states:

| State | Meaning | When |
|-------|---------|------|
| **ACTIVE** | Metric variance is above the noise floor | Training is still producing meaningful variation in this signal |
| **STALLED** | Metric variance has dropped below the noise floor | This signal has stopped changing meaningfully |
| **BOOTSTRAP** | Insufficient data for variance estimation | First $\leq$ 5 iterations |

---

## Δ-Metrics (Trends)

First differences of key metrics to measure per-iteration change.

| Metric | Formula | NaN condition |
|--------|---------|--------------|
| $\Delta$KL | $|\text{KL}_i - \text{KL}_{i-1}|$ | First iteration |
| $\Delta$Policy distance | $|\text{L1}_i - \text{L1}_{i-1}|$ | First iteration |
| $\Delta$Top1 | $|\text{Top1}_i - \text{Top1}_{i-1}|$ | First iteration |
| $\Delta$Drift | $|\text{Drift}_i - \text{Drift}_{i-1}|$ | First iteration |
| $\Delta$Grad norm | $|\text{Grad}_i - \text{Grad}_{i-1}|$ | First iteration |
| $\Delta$Param update | $\|\Delta\theta_i - \Delta\theta_{i-1}\|$ | First iteration |

**Cost**: $O(1)$ — difference between stored scalar values.

---

## Moving Averages

Rolling mean over windows of 5, 10, and 20 iterations for:

- `avg_loss`, `policy_loss`, `value_loss`
- `kl_mean`, `top1_pct`
- `drift_kl`
- `elo_candidate`
- `pred_entropy`

NaN until the window is fully populated (e.g., MA-20 → NaN for iterations 1–19).

**Cost**: $O(W)$ per metric where $W =$ window size. Ring buffer memory: $W \times 4$ bytes.

---

## Cost Summary

| Metric | Time Cost | Mem Cost | Frequency | Forward Passes |
|--------|-----------|----------|-----------|----------------|
| Policy/Value loss | $O(B \cdot A)$ | 4 B | Every iter | 0 (reused) |
| Gradient norm | $O(P)$ | 4 B | Every iter | 0 (reused) |
| Parameter update norm | $O(P)$ | $P \cdot 4$ B | Every iter | 0 (destructure) |
| Entropies | $O(A)$ | 8 B | Every iter | 0 (reused) |
| KL divergence | $O(B \cdot A)$ | $\sim N \cdot 4$ B | Every iter | 0 (reused) |
| Top-K agreement | $O(B \cdot A \cdot \ln A)$ | $\sim N \cdot 3$ B | Every iter | 0 (reused) |
| Policy L1 | $O(B \cdot A)$ | $\sim N \cdot 4$ B | Every iter | 0 (reused) |
| Root confidence | $O(1)$ | 4 B/pos | Every pos | 0 (MCTS) |
| Root Q | $O(1)$ | 4 B | Every iter | 0 (MCTS) |
| Search Gain | $O(1)$ | 4 B | Every iter | 0 (reuses root_q + network_value) |
| Network drift | $O(R \cdot A)$ | $R \cdot A \cdot 4$ B | Every iter | 1 forward ($R = 200$) |
| Value calibration | $O(B \ln B)$ | $\sim B \cdot 8$ B | Every iter | 0 (reused) |
| Elo | $O(1)$ | 24 B | Every iter | 0 |
| $\Delta$-metrics | $O(1)$ | 32 B | Every iter | 0 |
| Moving averages | $O(W)$ per metric | $W \cdot 4$ B each | Every iter | 0 |
| Convergence detection | $O(W)$ | $W \cdot 4$ B | Every iter | 0 |
| Health dashboard | $O(1)$ | 0 | Every iter | 0 |
| Diagnostic warnings | $O(1)$ | 0 | Every iter | 0 |
| JSONL write | $O(F) \sim 1$ KB | 1 KB flush buf | Every iter | 0 |

Where: $B =$ batch size, $A = 6$ (actions), $P =$ parameter count, $R = 200$ (reference set), $W = 20$ (window), $N =$ self-play positions per iteration.

Total overhead per iteration: $\sim 1$ KB JSONL write + 1 forward pass for drift ($R = 200$ states). No other forward passes added.
