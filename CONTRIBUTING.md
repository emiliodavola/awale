# Contributing to Awale RL

## Code Style & Conventions

### Docstring Convention (Julia)

Use Julia-style docstrings (`"""..."""`) with the following structure:

**Exported types** — document purpose and fields:

```julia
"""
    GameConfig

Configuration flags for Awale rule variants.

# Fields
- starvation::Symbol — :allow_capture or :prevent_starvation
- grand_slam::Symbol — :allow, :forbid, or :special
"""
```

**Exported functions** — document signature, purpose, arguments, and return value:

```julia
"""
    transition(state::GameState, action::Int) -> GameState

Apply `action` to `state`, returning a new immutable `GameState`.
Throws `ErrorException` if the action is not legal.
"""
```

**Internal helpers** — no docstring unless the logic is non-obvious or carries important invariants.

**Modules** — brief docstring at the top of each module describing its responsibility.

**Tone conventions:**

- English (code and technical artifacts are in English)
- Neutral tone, minimal markdown
- One-line summary, followed by details if applicable

## Releases

Releases are created manually through the `Release` GitHub Actions workflow
(`.github/workflows/release.yml`), which runs on `main` via `workflow_dispatch`.
The workflow never bumps versions, never commits, and never pushes to a branch:
it reads the version from `Project.toml` at dispatch time, runs the test suite
as a gate, and creates an annotated tag `v<version>` plus a GitHub Release with
auto-generated notes.

### Bumping the version (human PR)

1. In a normal feature branch targeting `main` (merged through the dev → main
   review flow), update both files:
   - `Project.toml` — the `version` field
   - `CITATION.cff` — the `version` and `date-released` fields
2. Open a pull request and merge it through the existing review flow so the
   new version lands on `main`.

The release workflow does not bump anything — the version is whatever
`Project.toml` says at dispatch time. All releases enter via a reviewed
dev → main PR; there are no hotfix-direct-to-main releases.

### Creating the release

1. Make sure the version bump PR is merged to `main`.
2. Go to *Actions* → *Release* → *Run workflow*, and run it against `main`.
   No inputs are required.
3. The workflow runs the full test suite (aborts if it fails), then creates
   and pushes the annotated tag `v<version>` and a GitHub Release with
   "What's Changed" notes generated since the previous tag.

If the tag was pushed but the GitHub Release creation fails (e.g. a transient
API error), the workflow leaves an orphan tag. Recovery: delete the remote tag
manually (`git push origin --delete v<version>`) and re-dispatch the workflow.

### Double-create guard

The workflow aborts with a clear error if the tag `v<version>` already exists
on the remote or a Release for that version already exists. Nothing is
overwritten — to release again, land a new bump PR first and re-dispatch.

### Token

No personal access token (PAT) is needed. The workflow declares
`permissions: contents: write` and uses the default `GITHUB_TOKEN`; its only
push is the tag ref, which branch protection on `main` does not block.