08_evaluation: Baselines, tournaments, and ELO estimation

Goals
- Provide deterministic evaluation pipelines for comparing models and baselines
- Support ELO-style ranking and regression testing against known positions

Baselines
- Random baseline: uniform legal move selection
- Heuristic baseline: implement small handcrafted evaluator (e.g., prefer captures, maximize immediate captures)
- Minimax baseline: exact minimax or alpha-beta with depth-limited search and heuristic evaluation

Evaluation harness
- Play round-robin tournaments with fixed RNG seeds and fixed time controls or simulation counts
- For each pairing, run N games with alternating colors to avoid bias
- Collect per-game metadata: seed, moves, leaf evaluations, time, model checkpoint id

ELO estimation
- Use standard Elo or Glicko2 implementation with deterministic tie-breaking
- Store per-match outcomes and update ratings offline (separate testing utility) to avoid stochastic update differences

Regression tests
- Maintain a set of canonical positions with expected best moves or scores; fail if model deviates beyond threshold

Testable properties
- Given same RNG and model checkpoints, evaluation match outcomes are identical
- Elo estimation code is deterministic given same match results file