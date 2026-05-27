09_testing: Property-based tests, unit tests, and test harness

Testing strategy
- Use specification-first approach: write property tests from specs in earlier directories before implementing logic.
- Test layers:
  1. Unit tests for pure functions (canonicalize, legal_actions, serialization)
  2. Property-based tests for invariants (seed conservation, non-negativity, idempotence)
  3. Integration tests for transition semantics across variants
  4. Determinism tests (fixed RNG seeds produce identical outputs)

Suggested tooling (Julia)
- Use Test.jl for unit tests
- For property-based testing use a QuickCheck-style library (e.g., QuickCheck.jl or PropertyTests.jl). If not available, implement small in-repo property test utilities with deterministic RNG.

Critical properties to test (executable forms)
- For many randomized/generated states s and legal action a:
  - seed conservation: sum(transition(s,a).board) + captured == 48
  - no negative pits
  - transition is pure: original s unchanged
  - serialization roundtrip equality
  - canonicalize idempotence
  - legal_actions only include pits with >0 seeds and respects starvation/forced-feeding rules

Determinism tests
- For deterministic network outputs stubbed to fixed logits and value, MCTS with fixed RNG produces identical visit distributions

Test data and fixtures
- Provide textual position fixtures in tests/fixtures/ for corner cases: starvation, grand-slam, immediate captures, long endgames, repetition scenarios

CI
- Ensure test suite runs on minimal Julia version pinned in Project.toml
- Run fast unit/property tests on PRs; run longer integration tests on release branch or nightly runs