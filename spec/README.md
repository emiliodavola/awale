# Specification directory for Awale RL project

This directory contains the current design contracts for the Awale research repo.

## Quick path

1. Read the numbered spec closest to the area you want to change.
2. Verify the code and tests still match that contract.
3. If the implementation changes the contract, update the spec in the same work unit.

## Structure

- `00_scope/`
- `01_game_rules/`
- `02_state_model/`
- `03_environment_api/`
- `04_neural_network/`
- `05_mcts/`
- `06_selfplay/`
- `07_training/`
- `08_evaluation/`
- `09_testing/`
- `10_performance/`
- `11_research/`

## Contract

- Core behavior specs should describe the **current implemented contract**.
- Future ambitions are fine, but they must be clearly labeled as future work.
- Specs are only authoritative when they are kept aligned with code and tests.
