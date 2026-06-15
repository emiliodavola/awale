06_selfplay: Self-play generation, replay buffer, and sample construction

Goals
- Produce high-quality training samples via MCTS-guided self-play
- Ensure reproducibility and deterministic sample generation when desired

Episode generation
- For each self-play episode:
  1. Initialize s0 from standard starting position (canonicalized)
  2. For each move t:
     - Run MCTS(root = canonicalize(s_t)) with fixed num_simulations and RNG
     - Derive policy_target π_t as visit_count_distribution over legal local actions (temperature schedule applied)
     - Select action a_t via: (a) sample from π_t if temperature>0; (b) argmax if eval mode
     - Store (s_t, π_t) in episode buffer
     - s_{t+1} = transition(s_t,a_t)
  3. On terminal, compute outcome z ∈ {-1,0,1} from perspective of player at s_t
  4. Emit training tuples (canonicalize(s_t), π_t, z) for all t

Replay buffer
- FIFO or prioritized buffer storing tuples with metadata: timestamp, seed, RNG state, provenance (self-play id)
- Support deterministic sampling via fixed RNG seed and shuffling algorithm

Temperature & exploration
- Temperature τ schedule: early moves τ>0 to encourage diversity; later moves τ→0 for deterministic play
- Inject Dirichlet noise at root during self-play only, controlled by RNG

Off-policy and checkpointing
- Tag samples with model-checkpoint id and training-step when generated
- Support replay across model updates; older samples may be down-weighted

Contracts
- selfplay_episode(config) must accept RNG seed and return reproducible episode given same seed and model checkpoint
- sample_batch(buffer, batch_size, rng)::Vector{(state,π,z)} deterministic given RNG seed

Testable properties
- Given fixed RNG and model outputs, self-play episode and emitted sample sequence are identical
- Replay buffer preserves seed/provenance metadata
- Temperature scheduling yields expected entropy patterns across moves