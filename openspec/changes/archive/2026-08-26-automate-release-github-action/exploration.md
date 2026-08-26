# Exploration: Automate the Software Release Process with a GitHub Action

## Current State

Release and versioning is **fully manual today**. There is no automation of any kind beyond the test CI. The current state is grounded in `git history`, `gh` inspection, and the workflow files.

### Version files (single source of truth)
- `Project.toml` — `version = "0.1.8"` (line 4). Package `Awale`, uuid `1fca9b84-4cc5-4472-afa3-0f8c8abc1f73`.
- `CITATION.cff` — `version: "0.1.8"` (line 6) and `date-released: "2026-07-31"` (line 7). Both fields must move together.
- Both are bumped together in a dedicated `release: bump version to X.Y.Z` commit (verified: commit `e138be5` touched `CITATION.cff` 2 lines and `Project.toml` 1 line).

### Existing CI
- Only one workflow exists: `.github/workflows/ci.yml` (name `CI`). It runs `test` on `ubuntu-latest`, Julia 1.11, instantiate + `Pkg.test()`. Triggers: `push` to `main` and `dev`, `pull_request` to `main` and `dev`, and `workflow_dispatch`. No release/tag workflow exists anywhere (confirmed across `origin/main` and `origin/dev`).
- `.github/PULL_REQUEST_TEMPLATE.md` requires a single `type:*` label and a `Closes #N` issue link.

### Observed release practice (from git + `gh`), for v0.1.8
1. Feature work lands in slices, each its own PR, e.g. #64 `feat(publication): model card helper functions (slice 1/3)`, #65 `(slice 2/3)`, #66 `(slice 3/3)`.
2. Spec sync PR to main: #67 `docs(spec): sync release-model-card spec delta from improve-hf-model-card`.
3. Version-bump PR: #68 `release: bump version to 0.1.8` — edits `Project.toml` + `CITATION.cff`.
4. Release-merge PR: #69 `release: v0.1.8 — model card rewrite and observability improvements` — merges the feature line into `main` (commit `0845d44`, a merge with parent `bf679ec` + `ecdf3ca`).
5. An **annotated git tag** `v0.1.8` is created on the main merge commit. Annotated tags exist for every version: `v0.1.0` … `v0.1.8`.
6. A **GitHub Release object is created for every tag** (verified via `gh release list`: v0.1.0..v0.1.8; v0.1.8 marked `Latest`, `draft: false`, `prerelease: false`). Release bodies use auto-generated "What's Changed" notes with PR links (e.g. v0.1.8 body "Model Card Rewrite and Observability", section "## What's Changed").
7. Optional post-training HF publication via `publish_hf.jl --publish --repo-id <ns>/awale-results` using `HF_TOKEN`.

### Branch conventions
- `main` and `dev` both exist; feature branches (`feat/*`, `fix/*`, `docs/*`, `chore/*`) branch off and target `dev`; releases go to `main`. CI gates both branches.
- Releases are tagged on `main`. The release merge PR brings `dev`'s accumulated changes into `main` and the annotated tag + Release object are created on that merge commit.

### HF publishing (relevant to release scope)
- `publish_hf.jl` (`--dry-run` / `--stage` / `--publish`) assembles a release bundle under `checkpoints/<architecture>/release/<release_id>/` from a **local `release_summary.toml`** produced by a completed training run, generates an English model card (`release_model_card()` in `src/Awale/Publication.jl`), and uploads safe `Float32` exports via `hf upload` (`HF_TOKEN` + `hf auth login` required).
- The HF publish is **data-dependent on local checkpoint artifacts on the training VM** — it is not reproducible from a CI checkout that lacks trained checkpoints. Model-card generation is governed by `openspec/specs/release-model-card/spec.md` (11 requirements); there is **no spec** governing the release/versioning process itself yet.

### Docs
- No `CHANGELOG.md`, no release-process doc. `CONTRIBUTING.md` covers only Julia docstring conventions (28 lines, no release/tag/version content). Release process is implicit in git history only.

## Affected Areas

- `Project.toml` — `version` field is the bump target.
- `CITATION.cff` — `version` and `date-released` fields are bump targets (date must update too).
- `.github/workflows/ci.yml` — existing test workflow; a release workflow should reuse its test logic (or be factored to a shared job).
- `.github/workflows/release.yml` (new) — the automation itself.
- `scripts/bump_version.jl` (new) — Julia-native script to bump `Project.toml` + `CITATION.cff` (respecting the repo's Julia/shell-only rule; no Python).
- `publish_hf.jl` / `src/Awale/Publication.jl` — optional HF publish stage; dependent on local checkpoints + `HF_TOKEN`, likely stays a manual/separate step.
- `openspec/specs/release-model-card/spec.md` — unaffected directly, but a new `release-automation` spec domain would be introduced by this change.

## Approaches

### 1. Manual-dispatch release workflow (recommended)
A new `.github/workflows/release.yml` triggered by `workflow_dispatch` with a semver input (e.g. `patch`/`minor`/`major`, or an explicit target version). On `main`:
- Run the existing test suite (reuse CI test steps).
- Run a Julia bump script (`scripts/bump_version.jl`) that updates `Project.toml` `version` and `CITATION.cff` `version` + `date-released`.
- Commit the bump as `release: bump version to X.Y.Z`, push to `main`.
- Create an annotated tag `vX.Y.Z`.
- Create a GitHub Release object with auto-generated "What's Changed" notes (matching the current convention).
- (Optional, separate) Trigger HF publish only if a staged bundle + `HF_TOKEN` secret are available.

- Pros: Matches the current explicit-bump convention (`release: bump version` commits); human decides the bump level (no surprise semver jumps); low complexity; reuses existing test job; fits the repo's manual-trigger workflow style.
- Cons: Bump level is not fully automatic (still a human decision at dispatch time).
- Effort: Medium.

### 2. Conventional-commits / commit-msg-driven semver
Parse merged commits since the last tag (e.g. `feat:` → minor, `fix:` → patch, breaking → major) and auto-derive the next version.

- Pros: Fully automatic; no human input at release time.
- Cons: The repo's commit style already uses conventional prefixes (`feat`, `fix`, `style`, `refactor`, `test`, `docs`) but semver auto-derivation adds parsing complexity and can surprise (a `feat` that should be a patch still bumps minor). Bigger behavior change vs. current explicit bumps.
- Effort: High.

### 3. Release-drafter + finalize dispatch (hybrid)
Use `release-drafter` to accumulate "What's Changed" notes into a draft release, then a manual `workflow_dispatch` finalizes with a chosen semver bump.

- Pros: Draft notes are continuously maintained; retains manual semver control; clean release notes.
- Cons: Adds a second workflow and more moving parts; heavier than needed for a single-maintainer repo that already writes release notes.
- Effort: High.

## Recommendation

Adopt **Approach 1**: a single `workflow_dispatch`-triggered `release.yml` on `main` that takes the semver bump as input, runs the test suite, bumps `Project.toml` + `CITATION.cff` via a Julia script (`scripts/bump_version.jl`), commits the bump with the established `release: bump version to X.Y.Z` convention, creates an annotated tag, and creates a GitHub Release with auto-generated notes — reproducing exactly what is done manually today.

- **Version-bump decision**: manual input at dispatch time (`patch`/`minor`/`major` or explicit version). This mirrors the current explicit-bump practice and avoids surprising semver jumps.
- **Test gating**: the workflow MUST run the full suite (`julia --project=. -e 'using Pkg; Pkg.test()'` after copying `config.toml.example`/`src/Awale/config.toml.example` → live configs, exactly as `ci.yml` does) and abort on failure before bumping.
- **CITATION.cff**: the bump script MUST update `date-released` in addition to `version` (both fields move together today).
- **HF publish**: keep it OUT of the core release workflow (or a separate optional dispatch). It depends on local checkpoint bundles on the training VM and `HF_TOKEN`; it is not reproducible from a CI checkout. Surface this as a deliberate scope boundary.

## Risks

- **GitHub Release objects already exist manually** (v0.1.0..v0.1.8, all with notes, v0.1.8 `Latest`). The workflow must reproduce that shape (annotated tag + Release with "What's Changed"), and must not double-create or conflict with an existing tag/release (guard on tag existence).
- **Token permissions**: creating a tag + Release needs `permissions: contents: write` on the workflow (and pushing the bump commit to `main` requires a token with push rights). The default `GITHUB_TOKEN` may need `permissions:` explicitly set; a PAT may be required for the push-to-main step.
- **HF publish needs credentials + artifacts**: `HF_TOKEN` secret and a staged `release_summary.toml` from a completed run on disk. CI has neither; do not wire HF publish into the release path unless artifacts are uploaded and a secret is provisioned. This is the main unknown for a fully end-to-end automated release.
- **Date field drift**: bumping only `version` in `CITATION.cff` and not `date-released` would leave the citation stale — the script must bump both.
- **No existing release spec**: `release-automation` is a new spec domain; the process is currently undocumented, so the spec must codify the observed convention (bump commit on main + annotated tag + Release notes + branch flow).
- **Dev→main merge**: the workflow runs on `main`; it must assume `dev` has already been merged/PR'd into `main` (the release merge PR #69 pattern) before the bump, so it does not try to release unreleased dev work.

## Ready for Proposal

Yes — the orchestrator should tell the user: the current release is fully manual (bump `Project.toml`+`CITATION.cff`, annotated tag, GitHub Release object, optional HF publish); a single `workflow_dispatch` release workflow with manual semver input, test gating, and a Julia bump script reproduces it; HF publish should be scoped out of CI unless checkpoints + `HF_TOKEN` are provisioned. Propose `automate-release-github-action`.
