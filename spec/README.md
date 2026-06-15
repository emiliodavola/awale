# Specification directory for Awale RL project

This directory contains formal specifications, invariants, contracts, testing strategies, and design constraints for a research-grade, specification-driven Awale (Oware) reinforcement learning system in Julia.

Structure (each folder contains a README.md following the template in this root):
- 00_scope/
- 01_game_rules/
- 02_state_model/
- 03_environment_api/
- 04_neural_network/
- 05_mcts/
- 06_selfplay/
- 07_training/
- 08_evaluation/
- 09_testing/
- 10_performance/
- 11_research/

Maintain these files as the authoritative source for design choices before any implementation. Each file must include goals, math definitions, invariants, contracts, edge cases, failure modes, and property-based test descriptions.