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