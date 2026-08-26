# Verification Report: automate-release-github-action

- **Change**: automate-release-github-action (release-automation delta)
- **Branch**: feat/automate-release-github-action → PR #71 (base `dev`, OPEN)
- **Mode**: Hybrid persistence (engram + openspec); Standard workflow (no Strict TDD — change introduces no Julia-logic units)
- **Date**: 2026-08-26

## Task Completion

| Task | Status | Evidence |
|---|---|---|
| 1.1 Scaffold (dispatch-only, main guard, contents:write, checkout@v4 fetch-depth 0) | complete | release.yml L1-24; actionlint exit 0 |
| 1.2 Inline test gate (setup-julia '1.11', cache@v2, config copies, instantiate, Pkg.test) | complete | release.yml L26-43 matches ci.yml verbatim |
| 1.3 Version extraction (TOML → GITHUB_OUTPUT) | complete | release.yml L45-50; local run outputs `0.1.8` |
| 1.4 Guards + create (tag guard, release guard, annotated tag, tag-only push, gh release create last) | complete | release.yml L52-76; both gh steps carry GH_TOKEN |
| 2.1 CONTRIBUTING.md Releases section | complete | CONTRIBUTING.md L44-89 |
| 3.1 actionlint | complete | Docker actionlint exit 0 (re-run in verify) |
| 3.2 E2E smoke (post-merge) | **deferred** | Requires GitHub runtime after merge; plan documented below |

## Executable Evidence

| Check | Command | Result |
|---|---|---|
| Workflow lint | `docker run --rm -v "${PWD}:/repo" -w /repo rhysd/actionlint:latest .github/workflows/release.yml` | exit 0, no errors/warnings |
| `branches` under `workflow_dispatch` probe | actionlint on temp workflow with `branches: [main]` | exit 1: `expected "inputs" key for "workflow_dispatch" section but got "branches"` — confirms apply-progress deviation claim |
| Version extraction | `julia -e 'using TOML; print(TOML.parsefile("Project.toml")["version"])'` | `0.1.8` (matches Project.toml) |
| ci.yml untouched | `git diff origin/dev -- .github/workflows/ci.yml` | empty |
| Diff scope | `git diff origin/dev --stat` | 8 files: release.yml (+76), CONTRIBUTING.md (+49), 6 openspec artifacts; no unintended files |
| Remote state (smoke grounding) | `git ls-remote --tags origin`, `gh release view v0.1.8` | tags v0.1.0..v0.1.8 exist; Release v0.1.8 exists ⇒ guards will fire |

## Spec Compliance Matrix

| # | Requirement | Scenario | Status | Evidence |
|---|---|---|---|---|
| 1 | Manual Dispatch Trigger | Dispatch on main | COMPLIANT | `on: workflow_dispatch` only; ref guard `github.ref == refs/heads/main` L14-19 |
| 1 | | No automatic trigger | COMPLIANT | no push/pull_request/schedule triggers |
| 2 | Version from Project.toml | Version read at dispatch | COMPLIANT | L45-50 TOML read; verified output 0.1.8; tag/release named from it |
| 3 | Bump via human PR | Bump lands via PR | COMPLIANT | CONTRIBUTING.md L53-64; no bump/commit steps in workflow |
| 4 | Test gate before release | Failing suite aborts | COMPLIANT | Pkg.test at L42-43 precedes all creates; step failure fails job |
| 4 | | Passing suite proceeds | COMPLIANT | gate passes ⇒ continues to guards/create |
| 5 | Annotated tag | Annotated tag on main | COMPLIANT | `git tag -a v<version> -m "Release v<version>"` L70 |
| 6 | GitHub Release w/ auto notes | Release with auto-generated notes | COMPLIANT | `gh release create v<version> --generate-notes` L76 |
| 7 | Double-create guard | Tag already exists | COMPLIANT | L52-57 ls-remote + `::error::` + exit 1 |
| 7 | | Release already exists | COMPLIANT | L59-66 `gh release view` + `::error::` + exit 1, GH_TOKEN set (L61) |
| 8 | Permissions & token | Default token sufficient | COMPLIANT | `permissions: contents: write` L6-7; GH_TOKEN=${{ github.token }} on both gh steps; no PAT |
| 8 | | No push step | COMPLIANT | only `git push origin v<version>` (tag ref) L71; CONTRIBUTING.md states no branch push |
| 9 | Explicit non-goals | HF publish stays out | COMPLIANT | no HF steps |
| 9 | | Existing tags untouched | COMPLIANT | workflow only touches v<version>; v0.1.0..v0.1.8 remain |
| 9 | | No automated version bump | COMPLIANT | no write to version files |

## Design Coherence

| Design decision | Implementation | Coherent? |
|---|---|---|
| Inline test gate (keep ci.yml untouched) | Inlined incl. `Pkg.instantiate` (design YAML omitted it; spec/task require it) | ✓ (improves on design) |
| TOML version extraction | TOML read + regex validation (L49) — added robustness | ✓ |
| gh CLI tag/release | matches | ✓ |
| Default GITHUB_TOKEN, no PAT | matches; GH_TOKEN added to release guard (design YAML omitted; needed for `gh release view` auth) | ✓ (fixes design omission) |
| dispatch-only, main-only | ref guard replaces invalid `branches` key (actionlint probe confirms rejection) | ✓ (correct deviation, intent preserved) |

## Issues

### CRITICAL
None.

### WARNING
- **W1 — Task 3.2 (E2E smoke) unchecked, deferred to post-merge.** Not a code defect: the workflow cannot be dispatched until PR #71 merges to `main`. Orchestrator-directed deferral; smoke plan below. No remediation on this branch.

### SUGGESTION
- **S1 — `fetch-depth: 0` is not strictly required** by the current steps (`git ls-remote`, `gh release create --generate-notes`, and `git tag -a` on HEAD all work from a shallow clone). Harmless and required by task 1.1; keep, no action.
- **S2 — `.github/.gitignore` (`*`) forces `git add -f` for future workflow files.** Consider a negation line (out of scope here) to avoid the force-add trap for future CI files.
- **S3 — Informational:** implementation improves two design-YAML omissions (`Pkg.instantiate` in the gate; `GH_TOKEN` on the release-exists guard) — no action.

## E2E Smoke Plan (task 3.2 — run AFTER merge to main)

**Precondition**: PR #71 merged to `main`; `git fetch origin`; confirm `.github/workflows/release.yml` present on `main`.

**Path A — duplicate-dispatch abort (current state)**
1. Actions → Release → Run workflow → branch `main` → Run (no inputs).
2. Expect: branch guard passes (`refs/heads/main`), checkout, julia setup, cache, config copies, instantiate, tests pass.
3. Expect: version read = `0.1.8`; **tag guard fires** `::error::Tag v0.1.8 already exists - aborting`, exit 1, job fails.
4. Verify: no new tag/release; `v0.1.8` tag + Release untouched (grounded: tag and Release confirmed to exist today).

**Path B — success run (after a future bump PR)**
1. Bump PR (dev→main, reviewed): edit `Project.toml` `version` → e.g. `0.2.0` and `CITATION.cff` `version` + `date-released`; merge.
2. Dispatch Release on `main`.
3. Expect: all steps green; annotated tag `v0.2.0` at main HEAD (`git show v0.2.0` shows "Release v0.2.0"); `git push origin v0.2.0` pushes **tag ref only** (verify no `refs/heads/` push in logs); `gh release create v0.2.0 --generate-notes` succeeds.
4. Verify: `git ls-remote --tags origin refs/tags/v0.2.0` present; Releases page shows v0.2.0 with "What's Changed" notes spanning v0.1.8..v0.2.0.
5. Re-dispatch `0.2.0`: expect clean abort by the tag guard (idempotency proof).
6. Failure recovery: if tag pushed but Release create fails → orphan-tag recovery per CONTRIBUTING.md (delete remote tag, re-dispatch).

## Risks
- **R1**: Tag push is the only remote write; if branch protection on `main` is later tightened to protect tag refs, tag push would need a PAT — documented assumption (design L16, CONTRIBUTING.md L88-89).
- **R2**: Smoke path A/B requires GitHub runtime and can only run post-merge; until then, guard behavior is verified by construction + live remote state.
- **R3**: `.github/.gitignore` (`*`) — future workflow files need `-f` (apply-progress risk, out of scope).

## Verdict
**PASS WITH WARNINGS** — all spec requirements/scenarios compliant, all core implementation tasks complete, no CRITICAL findings. Single WARNING is the orchestrator-deferred post-merge smoke (3.2), with a concrete execution plan.