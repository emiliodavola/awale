# Archive Report: improve-hf-model-card

**Date**: 2026-07-31
**Change**: improve-hf-model-card
**Issue**: #63 (CLOSED)
**Status**: Complete ✅

## Executive Summary

Full rewrite of the Hugging Face model card generator template in `src/Awale/Publication.jl`. Replaced the build-ID–titled, dead-bullet, path-leaking card with a stable-structured card providing real usage snippets, complete evaluation methodology, per-checkpoint promotion semantics, deduplicated bundle contents, real repo links, and shared metric rounding. Delivered as a 3-slice feature-branch chain (PRs #64, #65, #66) merged to `dev`.

## What Changed

| Area | Before | After |
|-------|--------|-------|
| Title | `# Awale release <id> model card` | `# Awale AlphaZero-like` (stable) |
| Code section | 4 dead plain-text bullets + bare URL | 5 hyperlinks into repo, repo link last |
| Source paths | `## Source paths` leaked local filesystem paths | Removed from public card |
| Selection flag | Single global `Selection promoted: false` line | Per-checkpoint narrative (never a global flag) |
| Bundle contents | `release_summary.toml` listed twice | Deduplicated, per-file descriptions |
| Metrics | Raw-precision (e.g., 61.702127659574465) | Rounded 2-4 sigdigits via `format_metric` |
| YAML front matter | Basic tags, unrounded values | Tags + `alphazero`, `self-play`, `board-game`; rounded values; metric descriptions |
| Evaluation | No baseline/methodology stated | RandomAgent baseline, 400 MCTS sims, 100 games, 56%-over-200 promotion gate |
| Generator version | 2 | 3 (forces restage of cached bundles) |
| Render robustness | Crashed on missing configs | Defensive fallbacks for older releases |

## Specs Synced

| Domain | Action | Details |
|--------|--------|---------|
| release-model-card | Updated | 9 ADDED, 2 MODIFIED, 1 RENAMED → 11 requirements total |

### Merge details
- **RENAMED first**: "Version Gate for Bundle Validity (Optional)" → "Version Gate for Bundle Validity"
- **MODIFIED**: "Code Section in Model Card" — replaced; old "Existing sections are unmodified" scenario (mandated `## Source paths` retention) is superseded
- **MODIFIED**: "Version Gate for Bundle Validity" — replaced; old v1→v2 semantics superseded by v2→v3
- **ADDED**: 9 new requirements appended (Stable Card Structure, Per-Checkpoint Promotion Semantics, Complete Evaluation Methodology, Shared Metric Rounding, Model-Index YAML Metadata, No Internal Paths in Public Card, Deduplicated Bundle Contents, Defensive Rendering of Older Releases, Test Coupling)
- Canonical spec restructured from delta format to flat requirement list

## Task Completion

| # | Task | Status | Notes |
|---|------|--------|-------|
| T1 | `format_metric` | [x] | Slice 1 (PR #64) |
| T2 | `read_bundle_configs` | [x] | Slice 1 (PR #64) |
| T3 | Parameter count helpers | [x] | Slice 1 (PR #64) |
| T4 | Caller-side config extraction | [x] | Slice 2 (PR #65) |
| T5 | Front matter rewrite | [x] | Slice 2 (PR #65) |
| T6 | Body rewrite + version bump | [x] | Slice 2 (PR #65) |
| T7 | Update flow test assertions | [x] | Slice 2 (PR #65) |
| T8 | Integration test coverage | [x] | Slice 3 (PR #66) |
| T9 | E2E restage edge cases | [x] | Slice 3 (PR #66) |
| T10 | Verify + commits | [x] | Reconciled at archive-time: verified inline — 780/780 green, JuliaFormatter clean |

**Reconciliation note**: T10 was unchecked in `tasks.md` but proven complete by apply-progress (#388): full suite 780/780 passing, formatter clean, all 3 PRs merged. Checkbox reconciled at archive-time.

## Delivery

| PR | Base | Branch | Tasks | Outcome |
|----|------|--------|-------|---------|
| #64 | dev | feat/improve-hf-model-card | T1-T3 (helpers) | Merged |
| #65 | feat/improve-hf-model-card | feat/improve-hf-model-card-2 | T4-T7 + 4R review fixes | Merged |
| #66 | feat/improve-hf-model-card-2 | feat/improve-hf-model-card-3 | T8-T9 (test depth) | Merged |

## Verification

- Full suite: 780/780 passing
- JuliaFormatter: clean (zero diff)
- No verify-report artifact; T10 verification was inline per orchestrator instruction

## Artifact Traceability

| Artifact | Engram Obs | OpenSpec Path |
|----------|-----------|---------------|
| Proposal | #381 | archive/2026-07-31-improve-hf-model-card/proposal.md |
| Spec delta | #382 | archive/2026-07-31-improve-hf-model-card/specs/release-model-card/spec.md |
| Design | #384 | archive/2026-07-31-improve-hf-model-card/design.md |
| Tasks | #385 | archive/2026-07-31-improve-hf-model-card/tasks.md |
| Apply-progress | #388 | (engram-only) |
| Archive report | #TBD | archive/2026-07-31-improve-hf-model-card/archive-report.md |

## Source of Truth Updated

`openspec/specs/release-model-card/spec.md` now reflects the new card template with 11 requirements:
1. Stable Card Structure
2. Code Section in Model Card (MODIFIED — no dead bullets, real links)
3. Per-Checkpoint Promotion Semantics
4. Complete Evaluation Methodology
5. Shared Metric Rounding
6. Model-Index YAML Metadata
7. No Internal Paths in Public Card
8. Deduplicated Bundle Contents
9. Defensive Rendering of Older Releases
10. Test Coupling
11. Version Gate for Bundle Validity (RENAMED + MODIFIED — mandatory v3)

## Risks

None. Version gate auto-invalidates cached bundles; older releases render defensively. Live Hub card updates on next publish only.

## SDD Cycle Complete

The change has been fully planned, implemented (3 PRs merged), verified (780/780 green), and archived. Ready for the next change.
