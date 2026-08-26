# Proposal: Automate the Release Process with a GitHub Action

## Intent

Release and versioning is fully manual today. Every release requires a human to: bump `Project.toml` `version` and `CITATION.cff` `version` + `date-released` in a `release: bump version to X.Y.Z` commit, push to `main`, create an annotated tag `vX.Y.Z`, and create a GitHub Release object with "What's Changed" notes. The process exists only implicitly in git history — no workflow, script, or spec codifies it. This change replaces that manual ceremony with a single `workflow_dispatch` GitHub Action that reproduces the exact current convention, eliminating copy/paste drift and missed steps.

## Scope

### In Scope
- New `.github/workflows/release.yml` — `workflow_dispatch` on `main`, semver input (`patch`/`minor`/`major` or explicit version).
- Test gate: run full suite (reuse `ci.yml` logic, incl. copying config examples) and abort before bumping on failure.
- New Julia-native `scripts/bump_version.jl`: bumps `Project.toml` `version` and `CITATION.cff` `version` + `date-released` (both move together).
- Commit bump as `release: bump version to X.Y.Z`, push to `main`.
- Create annotated tag `vX.Y.Z` + GitHub Release with auto-generated notes (guard against existing tag/Release).
- New spec domain `release-automation` codifying the convention.

### Out of Scope
- HF publish via `publish_hf.jl` (needs local checkpoints + `HF_TOKEN`; stays manual).
- Hotfix direct-to-main flow (all releases enter via PR dev→main).
- `CHANGELOG.md` file maintenance (notes come from auto-generated release notes; no changelog file is maintained).
- Migrating/back-filling existing tags v0.1.0..v0.1.8.

## Capabilities

> Contract between proposal and specs phases.

### New Capabilities
- `release-automation`: the version-bump + tag + GitHub Release workflow, its semver input, test gating, and the bump-script behavior.

### Modified Capabilities
- None (`release-model-card` is unaffected).

## Approach

Single `release.yml` on `main`, `workflow_dispatch` with a semver `input`. Steps: (1) `permissions: contents: write`; (2) run tests (reuse `ci.yml` config-copy + `Pkg.test()`, abort on failure); (3) run `scripts/bump_version.jl` to compute the new version from the input and update `Project.toml` + `CITATION.cff`; (4) commit `release: bump version to X.Y.Z` and push to `main`; (5) create annotated tag `vX.Y.Z`; (6) create GitHub Release with auto-generated notes. Edge cases: guard against double-creating tag/Release (abort if tag exists); a PAT may be required for the push-to-main step (default `GITHUB_TOKEN` may lack push rights).

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `.github/workflows/release.yml` | New | The release automation |
| `scripts/bump_version.jl` | New | Julia-native version-bump script |
| `.github/workflows/ci.yml` | Modified | Reused/factored for the test gate (no behavior change) |
| `openspec/specs/release-automation/` | New | New spec domain |
| Docs (e.g. `CONTRIBUTING.md`) | Modified | Document the release workflow |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Double-creating tag/Release for existing version | Med | Guard on tag/Release existence, abort before creating |
| Push-to-main fails with default token | Med | Set `permissions: contents: write`; document PAT fallback |
| `CITATION.cff` date drift (only version bumped) | Low | Script bumps `date-released` alongside `version` |
| Releasing unreleased dev work | Low | Workflow runs on `main` only, assumes dev already merged |

## Rollback Plan

Delete the tag and Release via `gh release delete vX.Y.Z --yes` and `git push origin --delete vX.Y.Z`; revert the bump commit with `git revert` (or `git reset` + force-push if unpushed) to restore the prior version. The script only edits the two version files, so recovery is a one-commit revert.

## Dependencies

- Julia ≥ 1.11 available on the runner (same as `ci.yml`).
- Repository write permissions for tag/Release/push; PAT if push-to-main requires it.

## Success Criteria

- [ ] `workflow_dispatch` with a semver input produces a bump commit, annotated tag, and GitHub Release on `main`.
- [ ] Test gate aborts the run (no tag/Release) when the suite fails.
- [ ] Running for the current version aborts cleanly (no duplicate tag/Release).
- [ ] `CITATION.cff` `date-released` updates with the release date.
- [ ] Release notes auto-generated between previous and new tag.
