# Archive Report: automate-release-github-action

**Date**: 2026-08-26
**Change**: automate-release-github-action
**Branch**: feat/automate-release-github-action
**PR**: #71 (base `dev`, OPEN — merging is a separate delivery step)
**Status**: Complete (intentional-with-warnings) ✅

## Executive Summary

Replaced the fully-manual release process (bump `Project.toml` + `CITATION.cff`, annotated tag `vX.Y.Z`, GitHub Release object) with a single dispatch-only GitHub Action. A new `.github/workflows/release.yml` runs the full test suite as a gate, reads the current version from `Project.toml` (the single source of truth), guards against an existing tag/Release, creates an annotated tag `v<version>` (pushing only the tag ref), and creates a GitHub Release with auto-generated "What's Changed" notes. The version bump itself stays a human-reviewed dev→main PR (documented in a new CONTRIBUTING.md "Releases" section); the workflow never bumps, commits, or pushes to a branch. Implemented as a single PR (#71, ~150 lines), verified PASS-with-warnings, delivered on `feat/automate-release-github-action`.

## What Changed

| Area | Before | After |
|------|--------|-------|
| Release process | Fully manual: bump commit + annotated tag + Release object, codified only in git history | Single `workflow_dispatch` GitHub Action (`.github/workflows/release.yml`) on `main`, no inputs |
| Version bump | Manual `release: bump version to X.Y.Z` commit + push | Human-reviewed dev→main PR editing `Project.toml` `version` + `CITATION.cff` `version`/`date-released`; workflow does NOT bump |
| Test gating | None for releases | Full suite inlined as gate (`Pkg.test()`, ci.yml steps); failure aborts before any tag/Release |
| Tag/Release | Manual `git tag -a` + `gh release create` | Automated: annotated tag + tag-ref-only push + `gh release create --generate-notes` |
| Duplicate protection | None | Double-create guard (tag via `ls-remote`, Release via `gh release view`) aborts with `::error::` |
| Token | Manual (human token) | `permissions: contents: write`; default `GITHUB_TOKEN` suffices (no branch push, so no PAT) |
| Docs | No release-process doc | CONTRIBUTING.md "Releases" section |
| Spec | No release/versioning spec | New `release-automation` spec domain (9 requirements) |

## Specs Synced

| Domain | Action | Details |
|--------|--------|---------|
| release-automation | Created | 9 ADDED requirements → 9 requirements total (new spec domain, canonical flat format) |

### Merge details
- `openspec/specs/release-automation/` did not exist — this is a NEW capability, so the delta spec (written as ADDED requirements) became the full main spec at `openspec/specs/release-automation/spec.md`.
- Canonical spec restructured from delta format to the repo's flat requirement list (`# release-automation` + `## Requirements` + `### Requirement:` blocks), matching `openspec/specs/release-model-card/spec.md`.
- Requirements synced: Manual Dispatch Trigger, Version Sourced from Project.toml, Version Bump Is a Human-Reviewed PR, Test Gate Before Release, Annotated Tag Creation, GitHub Release Creation, Double-Create Guard, Permissions and Token, Explicit Non-Goals.

## Task Completion

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1.1 | Workflow scaffold (dispatch-only, main guard, contents:write, checkout@v4) | [x] | release.yml L1-24; actionlint exit 0 |
| 1.2 | Inline test gate (ci.yml steps incl. `Pkg.instantiate`) | [x] | release.yml L26-43; ci.yml untouched (`git diff` empty) |
| 1.3 | Version extraction (TOML → GITHUB_OUTPUT) | [x] | release.yml L45-50; local run outputs `0.1.8` |
| 1.4 | Guards + create (tag/release guards, annotated tag, tag-only push, gh release create) | [x] | release.yml L52-76 |
| 2.1 | CONTRIBUTING.md "Releases" section | [x] | CONTRIBUTING.md L44-89 |
| 3.1 | actionlint validation | [x] | Docker actionlint exit 0 (re-run in verify) |
| 3.2 | E2E smoke (post-merge) | [ ] | **Deferred** — requires GitHub runtime after PR #71 merges to `main` |

**Intentional-with-warnings note (deferred task)**: Task 3.2 (E2E smoke) is NOT a stale checkbox for completed work — it is a genuine post-merge GitHub runtime step that cannot execute until PR #71 merges to `main`. The orchestrator explicitly directed archive to record the change state with merging and post-merge smoke as a separate delivery step. Verify-report is PASS-with-warnings (no CRITICAL). Full smoke plan (Path A duplicate-abort, Path B success run) is preserved in the archived `verify-report.md`.

## Delivery

| PR | Base | Branch | Tasks | Outcome |
|----|------|--------|-------|---------|
| #71 | dev | feat/automate-release-github-action | 1.1-1.4, 2.1, 3.1 | **OPEN** (single PR, ~150 lines; merge to `dev` then `main` pending) |

Commits (work units): `ba5e404` (SDD artifacts), `a6ab75e` (scaffold), `8fdc984` (inline gate), `7364596` (guards + tag/release), `512b5c9` (docs), `64e5a5f` (apply-progress), `7a9ecc5` (gh auth + version validation).

## Verification

- actionlint (`rhysd/actionlint:latest`, Docker): exit 0, no errors/warnings (re-run in verify).
- Version extraction tested locally under `bash`: `0.1.8`.
- Spec compliance matrix: 15/15 scenarios COMPLIANT (see verify-report.md).
- Design coherence: implementation improves two design-YAML omissions (`Pkg.instantiate` in gate; `GH_TOKEN` on release-exists guard).
- No CRITICAL issues. One WARNING (W1 — task 3.2 deferred). Three suggestions (S1 fetch-depth, S2 `.gitignore` force-add, S3 informational).
- **Verdict: PASS WITH WARNINGS.**

## Artifact Traceability

| Artifact | Engram Obs | OpenSpec Path |
|----------|-----------|---------------|
| Exploration | #579 | archive/2026-08-26-automate-release-github-action/exploration.md |
| Proposal | #580 | archive/2026-08-26-automate-release-github-action/proposal.md |
| Spec delta | #581 | archive/2026-08-26-automate-release-github-action/specs/release-automation/spec.md |
| Design | #583 | archive/2026-08-26-automate-release-github-action/design.md |
| Tasks | #588 | archive/2026-08-26-automate-release-github-action/tasks.md |
| Apply-progress | #590 | archive/2026-08-26-automate-release-github-action/apply-progress.md |
| Verify-report | #594 | archive/2026-08-26-automate-release-github-action/verify-report.md |
| Archive report | (this observation) | archive/2026-08-26-automate-release-github-action/archive-report.md |

## Source of Truth Updated

`openspec/specs/release-automation/spec.md` now reflects the new release automation with 9 requirements:
1. Manual Dispatch Trigger
2. Version Sourced from Project.toml
3. Version Bump Is a Human-Reviewed PR
4. Test Gate Before Release
5. Annotated Tag Creation
6. GitHub Release Creation
7. Double-Create Guard
8. Permissions and Token
9. Explicit Non-Goals

## Risks

- **R1**: Tag push is the only remote write; if `main` branch protection is later tightened to protect tag refs, tag push would need a PAT — documented assumption (design L16, CONTRIBUTING.md).
- **R2**: Post-merge smoke (3.2) remains pending; until then guard behavior is verified by construction + live remote state (tags v0.1.0..v0.1.8 and Release v0.1.8 confirmed to exist).
- **R3**: `.github/.gitignore` (`*`) forces `git add -f` for future workflow files — out of scope.
- **R4 (delivery)**: PR #71 is OPEN, not merged. The release workflow is not active until it merges to `main`; the post-merge smoke must run then.

## SDD Cycle Complete

The change has been fully planned, implemented (PR #71), verified (PASS-with-warnings), and archived. Remaining delivery steps (separate from archive): merge PR #71 (dev→main) and run the post-merge E2E smoke per the plan in verify-report.md.
