# 01_game_rules: Formal game rules and variant configuration

## Goals

- Precisely state the 2-row, 12-pit Awalé variant implemented by the repo.
- Keep terminal-state behavior explicit so training, search, and evaluation agree.
- Preserve deterministic, testable rule branches for all supported variant flags.

## Core definitions

- Board vector: `b ∈ N^12`.
- Player turn: `p ∈ {1,2}`.
- Seed conservation invariant: `sum(b) + captured_1 + captured_2 == 48`.
- Canonical state view is defined in `spec/02_state_model` and is used for learning inputs and repetition hashing.

## Configurable variant flags

The active ruleset must be explicit in `GameConfig`.

- `starvation`: `:allow_capture` | `:prevent_starvation`
- `grand_slam`: `:allow` | `:forbid` | `:special`
- `repetition`: `:draw_on_repeat` | `:revert` | `:score_diff`
- `forced_feeding`: `:allow_move_feeding` | `:require_feed`

## Terminal-state contract

The canonical ruleset must terminate the game when any of the following applies:

1. **Capture threshold win**
   - A player wins immediately when their captured total reaches **25 or more** seeds.
   - Since the board contains 48 seeds total, this makes the result mathematically unreachable for the opponent.

2. **24–24 draw**
   - If both players have captured **24 seeds**, the game ends in a draw.

3. **No-feed terminal state**
   - If the player to move has no seeds on their side and the opponent has no legal move that can feed them, the game ends immediately.
   - Each player captures the seeds remaining on their own side before the final result is computed.

4. **Repetition / cycle handling**
   - Repetition is detected from the canonical history hash.
   - `:draw_on_repeat` resolves the game as a draw.
   - `:score_diff` resolves the game by final captured totals.
   - `:revert` is not a terminal resolution; the caller is responsible for applying revert semantics.

## Precise move semantics

- Legal actions are the 6 pits on the current player's side that contain at least one seed.
- Actions are exposed locally as `1..6`; the environment maps them to absolute pit indices.
- Sowing is counterclockwise and must preserve seed conservation.
- Capture follows the classic rule: when the last seed lands in an opponent pit with `2` or `3` seeds, capture those seeds and continue backward while the prior opponent pits also contain `2` or `3`.

## Edge cases and special handling

- **Starvation**: when `:prevent_starvation` is active, starving moves are filtered out when non-starving alternatives exist.
- **Forced feeding**: when `:require_feed` is active and the opponent is starved, legal actions are filtered to feeding moves when any exist.
- **Grand slam**: full-capture handling must remain explicit and deterministic, even if the current implementation keeps it permissive.
- **Repetition**: full-turn hashes must be tracked so the configured repetition policy can be applied deterministically.

## Failure modes

- Ill-formed board vectors
- Actions outside legal range
- Variant flags unset or contradictory
- Terminal detection that disagrees with the selected ruleset

## Testable properties

- Captures never create negative pit counts
- Sowing + capture preserves seed conservation
- Repetition detection matches serialization-based hashing
- `legal_actions` only returns moves that are legal under the active variant flags
- Terminal detection is consistent with the 25-seed, 24–24, no-feed, and repetition rules
