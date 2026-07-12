01_game_rules: Formal game rules and variant configuration

Goals
- Precisely state rules for the 2-row, 12-pit Awale variant with configurable options.
- Provide mathematically precise rule descriptions and pseudocode for all rule branches.

Core definitions
- Board vector: b ∈ N^{12}. Indexing convention: 0..5 belong to Player 1 (current-player view canonicalization applies), 6..11 to Player 2. Implementation may choose canonical indexing; spec requires mapping functions.
- Total seeds S = sum(b) + captured_1 + captured_2; invariant S == 48 at game start.
- Turn: p ∈ {1,2}.

Configurable variant flags (must be explicit in GameState):
- starvation: :allow_capture (default true) | :prevent_starvation
- grand_slam: :allow | :forbid | :special_rules (describe)
- repetition: :draw_on_repeat | :revert_to_previous | :score_diff
- forced_feeding: :allow_move_feeding | :require_feed

Precise move semantics
- Action space: select one of the 6 pits on current player's side containing >0 seeds. Represent action as integer a ∈ {0..5} relative to player's side; environment maps to absolute pit index.
- Sowing: counterclockwise distribution of seeds, skipping no pits unless variant demands.
- Capture rule (classic): when last seed lands in opponent pit with k ∈ {2,3} seeds (after sowing), capture those k; continue capturing consecutively backwards while previous opponent pits contain 2 or 3.

Edge cases and special handling
- Starvation: define behaviors for moves that would leave opponent with zero legal moves. In :prevent_starvation mode, moves that exclusively starve opponent are illegal unless they capture seeds.
- Grand slam: define full-capture detection (player captures all opponent seeds in a turn) and configured handling.
- Repetition: repository will record full-turn hashes; repetition handling strategy is configurable.

Failure modes
- Ill-formed board vectors
- Actions outside legal range
- Variant flags unset or contradictory

Testable properties
- Captures never create negative pit counts
- Sowing + capture → seed conservation (see spec/02_state_model)
- Repetition detection consistency with serialization-based hashing