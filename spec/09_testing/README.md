# 09_testing: Current test strategy

This spec documents the current test layout and the gaps that still exist.

## Quick path

1. Run `julia --project=. -e 'using Pkg; Pkg.test()'` for the package-style test path used by CI.
2. Use `julia --project=. test/runtests.jl` for local direct execution.
3. Keep new behavior covered in the same commit as the code change.

## Current test layers

| Layer | Current coverage |
|---|---|
| State/model | serialization, canonicalization, encoding, model-head expectations |
| Environment | legal actions, transitions, reward/terminal semantics |
| Invariants | seed conservation across repeated random transitions |
| Variants | forced-feeding and starvation-rule filtering |
| Search/training | PUCT perspective, deterministic eval mode, replay/training iteration, checkpoint policy, entrypoint loading |

## Current CI contract

- CI runs `Pkg.test()`.
- The package test path must not depend on the current working directory.
- Top-level scripts and model config resolution must therefore be repo-relative.

## Current gaps

- No dedicated `test/fixtures/` directory yet.
- No full optimizer/RNG continuation test for checkpoints.
- No canonical-position regression suite yet.
- Variant coverage is present but still small.

## Determinism expectations

- Search determinism tests should use fixed RNG seeds.
- Invariant/property-style tests should prefer explicit seeds when randomness is used.
- Behavior-level tests are preferred over helper-only assertions when validating user-visible scripts.

## Testing checklist

- [ ] `Pkg.test()` passes
- [ ] `test/runtests.jl` passes
- [ ] invariants remain enforced
- [ ] rule variants stay covered in the active suite
- [ ] top-level training/evaluation scripts load correctly under tests
