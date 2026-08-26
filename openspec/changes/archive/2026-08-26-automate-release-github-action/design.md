# Design: Automate Release with a GitHub Action (release-automation) — Modelo B

## Technical Approach

A single dispatch-only workflow (`.github/workflows/release.yml`) on `main` with **no inputs**. It reads the current version from `Project.toml` (single source of truth, already bumped by a human PR), runs the existing test suite as a gate, guards against an existing tag/Release, creates an annotated tag `v<version>` on the current `main` commit, pushes **only the tag ref**, and creates a GitHub Release with auto-generated notes. The workflow never bumps, never commits, and never pushes to a branch. The version bump itself is a normal, reviewed dev→main PR editing `Project.toml` + `CITATION.cff` (documented in CONTRIBUTING.md).

This replaces the rejected Modelo A (bump input + `bump_version.jl` + CI commit/push to `main` + PAT fallback) entirely — none of that approach is preserved.

## Architecture Decisions

| Decision | Choice | Alternatives | Rationale |
|----------|--------|-------------|-----------|
| Test gate | **Inline** the 4–5 ci steps (setup-julia, cache, config copies, instantiate, `Pkg.test()`) in release.yml | `workflow_call` into `ci.yml` | `workflow_call` requires editing the always-on CI file (`on:`), risking the production PR/push gate for a rarely-run release job. Inlining duplicates ~5 steps but keeps `ci.yml` untouched and the release self-contained — the safer tradeoff for a deterministic repo. |
| Version extraction | `julia -e 'using TOML; print(TOML.parsefile("Project.toml")["version"])'` | `grep`/`awk` on the version line | Repo rule prefers Julia-native tooling; TOML parsing is robust to spacing/quoting, no regex fragility. Requires `setup-julia` (already present for the test gate) — no extra dependency. |
| Tag/Release creation | `gh` CLI (preinstalled on runners): `git tag -a` + `git push origin vX.Y.Z`; `gh release create --generate-notes` | `softprops/action-gh-release` | Zero third-party deps; `--generate-notes` reproduces the observed "What's Changed" convention; ambient `GITHUB_TOKEN`. |
| Token | `permissions: contents: write`; default `GITHUB_TOKEN` only, no PAT | PAT fallback (Modelo A) | No branch push happens — only a **tag** push, which `GITHUB_TOKEN` with `contents: write` permits even under branch protection on `main`. Matches the revised spec's "no PAT" requirement. |
| Version handoff | `TOML` read → step output → tag/release name | Input param | No input exists; `Project.toml` is the single source of truth at dispatch time (spec requirement). |

## Workflow Design (`.github/workflows/release.yml`)

```yaml
name: Release
on:
  workflow_dispatch:          # manual only, no inputs, no semver
permissions:
  contents: write             # enough for tag + release; NO branch push
jobs:
  release:
    runs-on: ubuntu-latest    # matches ci.yml; avoids Windows CRLF/bash pitfalls
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with: {fetch-depth: 0}   # full history for auto-generated release notes

      - name: Setup Julia
        uses: julia-actions/setup-julia@v2
        with: {version: '1.11'}
      - name: Cache Julia artifacts
        uses: julia-actions/cache@v2
      - name: Prepare runtime configs
        run: |
          cp config.toml.example config.toml
          cp src/Awale/config.toml.example src/Awale/config.toml
      - name: Run tests (gate)   # aborts the run if this fails
        run: julia --project=. -e "using Pkg; Pkg.test()"

      - name: Read version from Project.toml
        id: version
        run: echo "version=$(julia -e 'using TOML; print(TOML.parsefile(\"Project.toml\")[\"version\"])')" >> "$GITHUB_OUTPUT"

      - name: Guard tag exists
        run: |
          if git ls-remote --tags origin "refs/tags/v${{ steps.version.outputs.version }}" | grep -q .; then
            echo "::error::Tag v${{ steps.version.outputs.version }} already exists — aborting"; exit 1; fi
      - name: Guard release exists
        run: |
          if gh release view "v${{ steps.version.outputs.version }}" >/dev/null 2>&1; then
            echo "::error::Release v${{ steps.version.outputs.version }} already exists — aborting"; exit 1; fi

      - name: Annotated tag (ONLY push — to tag ref, not a branch)
        run: git tag -a "v${{ steps.version.outputs.version }}" -m "Release v${{ steps.version.outputs.version }}" && git push origin "v${{ steps.version.outputs.version }}"

      - name: Create GitHub Release
        env: {GH_TOKEN: ${{ github.token }}}
        run: gh release create "v${{ steps.version.outputs.version }}" --generate-notes
```

Notes: pushing a **tag ref** (`refs/tags/*`) is allowed with the default token under `contents: write`, and is not blocked by branch protection on `main` (that protects `refs/heads/*`). This is the workflow's only push. `gh` is preinstalled on `ubuntu-latest`. Tag/Release pushes do not re-trigger CI, so no loop.

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `.github/workflows/release.yml` | Create | Dispatch-only release pipeline (above). |
| `.github/workflows/ci.yml` | None | **Unchanged** — test steps are inlined; no `workflow_call` edit. |
| `CONTRIBUTING.md` | Modify | New "Releases" section (PR bump convention + dispatch steps). |

No `scripts/bump_version.jl` is created; no `test/test_bump_version.jl` exists; nothing new is added to `test/runtests.jl`.

## PR Bump Convention (CONTRIBUTING.md "Releases")

Document the human workflow: (1) in a normal feature/branch PR targeting `main` (dev→main merge flow), edit `Project.toml` `version` and `CITATION.cff` `version` + `date-released`; (2) merge through the existing review flow so the new version lands on `main`; (3) dispatch the `Release` workflow manually (`workflow_dispatch`). State explicitly: the workflow does NOT bump, commit, or push to a branch — the version is whatever `Project.toml` says at dispatch time. All releases enter via a reviewed dev→main PR (no hotfix-direct-to-main).

## Data Flow

```
human PR (dev→main): bump Project.toml + CITATION.cff ──▶ merged on main
workflow_dispatch ──▶ tests (gate; fail ⇒ abort) ──▶ read version from Project.toml
  ──▶ guard tag (ls-remote) ──▶ guard release (gh release view)
  ──▶ git tag -a v<version> ──▶ push tag ref only ──▶ gh release create --generate-notes
```

## Testing Strategy

| Layer | What | Approach |
|-------|------|----------|
| Workflow YAML | Syntax/structure of `release.yml` | `actionlint` in the verify phase (no Julia unit tests — no script exists). GitHub also validates on push. |
| Local unit | None | Nothing testable locally; no new script/module is introduced. |
| E2E | Tag + Release creation | Manual dispatch against `main` in a real run (verify phase smoke). |

## Migration / Rollout

No migration. Historical tags `v0.1.0`..`v0.1.8` and their Releases are untouched (non-goal). First Modelo B release uses the version already on `main` at dispatch.

## Failure Modes

| Point | Behavior |
|-------|----------|
| Tests fail | Test gate step fails → job aborts before tag/Release; nothing created. |
| Tag `v<version>` exists | Guard emits `::error::Tag vX.Y.Z already exists — aborting` and exits 1. |
| Release `v<version>` exists | Guard emits `::error::Release vX.Y.Z already exists — aborting` and exits 1. |
| Dispatch before bump merged | `Project.toml` still holds the previous version → the guard (or `gh release create`) aborts; operator bumps via PR first. |
| Tag pushed, Release create fails | `gh release create` fails; operator either runs `gh release create` manually or deletes tag (`git push origin --delete vX.Y.Z`) and re-dispatches. |

## Open Questions

- None blocking. Whether `main` has branch protection is irrelevant to the tag push (tag refs are not branch-protected), so no PAT is ever required.
