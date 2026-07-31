# Design: Improve Hugging Face Model Card Quality

## Technical Approach

Cohesive rewrite of the card render path in `src/Awale/Publication.jl` (approach 1 from exploration). Two functions — `release_model_card` (L587-668) and `write_model_card_front_matter` (L177-219) — are restructured into a pure body builder plus small section helpers; config parsing and file-size IO are extracted to the caller (`write_release_bundle`); one shared rounding helper feeds both the body and the YAML `model-index`; `MODEL_CARD_GENERATOR_VERSION` bumps 2→3, invalidating cached bundles so restage is automatic via the existing `bundle_is_valid` gate. Tests update in the same change. Satisfies the spec delta (9 ADDED, 2 MODIFIED, 1 RENAMED).

## Architecture Decisions

### D1: Pure render path; config IO in the caller

| Option | Tradeoff | Decision |
|---|---|---|
| Parse TOMLs inside `release_model_card` | Mixes IO into pure render; old bundles may lack files | ✗ |
| Extract configs in `write_release_bundle`, pass dicts in | Render stays pure/testable; one defensive helper; spec mandate | ✓ |

`release_model_card` gains keyword args `training_config=Dict{String,Any}(), model_config=Dict{String,Any}(), model_params=nothing` — defaults render old releases defensively.

### D2: Shared rounding helper

| Option | Tradeoff | Decision |
|---|---|---|
| `format_metric(x::Real)::String` — `round(x; sigdigits=4)` then strip trailing zeros | 2-4 sig decimals: 61.702…→"61.7", 71.0→"71", 0.42→"0.42"; single source, no body/YAML drift | ✓ |
| Per-site formatting | Drift risk (spec scenario demands agreement) | ✗ |

Raw values are never printed; body and YAML both call `format_metric`.

### D3: Per-checkpoint promotion narrative

Derived from optional `metrics.selection_promoted`; the global flag is never printed as a line:
- `true` → "model_best passed the promotion gate when selected".
- `false` → "model_best did not pass the promotion gate at the last selection".
- absent → neutral: "best-scoring checkpoint during training".
- `model_last`/`model_final` → always "final run state; not subject to the promotion gate".

### D4: Parameter count

`public_model_parameter_count(bundle_dir)` = `filesize(artifacts/model_best.f32) ÷ 4` when `model_export_format == "float32"` (verified: 126236B → 31559); else `model_parameter_count(model_config)` summing Dense `in*out+out` and Conv `prod(kernel)*in*out+out`; else `nothing` → card shows "not recorded". Computed in the caller (file IO), passed as `model_params`.

### D5: Deduplicated Bundle Contents

Root cause: L652 prints `release_summary.toml` explicitly, then the L655 loop re-lists `artifact_specs` keys, which already contain it (L776). Fix: one ordered list = sorted `artifact_specs` keys (each with a description keyed by filename) + `manifest.toml`/`README.md` appended only if absent.

### D6: Stable structure; release ID moves to metadata

Title `# Awale AlphaZero-like`. Release ID, commit SHA, timestamp, bundle kind, export format move to a compact `## Release` section (keeps L194-195 assertions valid). Section order per spec: Model Details → Usage → Training Details → Evaluation → Limitations → Bundle contents → Code (real links) → Citation + License. YAML `model-index` name = stable model name; tags gain `alphazero`, `self-play`, `board-game`; each custom metric gets a `description`.

## Data Flow

```
write_release_bundle(bundle_dir, summary, artifact_specs; bundle_kind, model_export_format)
  ├─ read_bundle_configs(bundle_dir)                  # IO: artifacts/training_config.toml + model_config.toml; try/catch → Dict()
  ├─ model_params = public_model_parameter_count(...) # IO: filesize
  └─ write_release_model_card(...) → release_model_card(...)   # pure, no IO
       ├─ write_model_card_front_matter(io, ...)      # YAML, values via format_metric
       └─ body: card_* section helpers (println to io)
  README.md ── publish_model_card_command ──→ HF repo root
```

Section helpers (same file, existing IOBuffer style): `card_model_details`, `card_usage`, `card_training_details`, `card_evaluation`, `card_limitations`, `card_bundle_contents`, `card_code_section`, `card_citation`.

## File Changes

| File | Action | Description |
|---|---|---|
| `src/Awale/Publication.jl` | Modify | `MODEL_CARD_GENERATOR_VERSION` 2→3; rewrite card body + front matter; add `format_metric`, `read_bundle_configs`, `model_parameter_count`, `public_model_parameter_count`, section helpers |
| `test/test_publication_flow.jl` | Modify | L90-107 and L194-195 assertions; new testsets below |
| `openspec/changes/improve-hf-model-card/design.md` | Create | This document |

## Interfaces / Contracts

```julia
format_metric(x::Real)::String                               # 2-4 sig decimals, trailing zeros stripped
read_bundle_configs(bundle_dir)::Tuple{Dict,Dict}            # defensive: try/catch → empty dicts
model_parameter_count(model_config::Dict)::Union{Int,Nothing}
public_model_parameter_count(bundle_dir; model_export_format)::Union{Int,Nothing}
release_model_card(summary, artifact_specs; bundle_kind, model_export_format,
                   training_config=Dict{String,Any}(), model_config=Dict{String,Any}(),
                   model_params=nothing)::String              # pure render; no new IO
```

## Testing Strategy

| Layer | What | Approach |
|---|---|---|
| Unit | `format_metric` | 61.702127659574465→"61.7"; 71.0→"71"; 0.42→"0.42" |
| Unit | `read_bundle_configs` | missing files → empty dicts, no throw |
| Unit | `model_parameter_count` | `src/Awale/config.toml` MLP variant → 31559 |
| Integration | Card text (L90-107) | stable title, all sections, `!occursin("Source paths")`, `!occursin("Selection promoted")`, no `checkpoints\` path, `## Code` links with repo last, `Release ID:` present |
| Integration | L194-195 | `Bundle kind:` / `Model export format:` still present (Release section) |
| Integration | Dedup | `count("release_summary.toml", card) == 1` |
| Integration | Rounding agreement | seed raw 61.702…; body shows `61.7` AND YAML `value: 61.7` |
| Integration | Promotion semantics | seed `selection_promoted` true and false; assert per-checkpoint wording, no global flag |
| Integration | Defensive render | `release_model_card` with empty config dicts and summary missing optional metrics → renders with fallbacks |
| E2E | Restage | existing version-change testset; add: manifest missing `model_card_generator_version` → `bundle_is_valid == false` → restaged |

## Migration / Rollout

Version bump auto-invalidates cached bundles. Confirmed from code (L900-902): `get(manifest, "model_card_generator_version", nothing)` fails the `isa Integer` check for missing or non-integer values → `bundle_is_valid` returns `false` → `stage_release_bundle` rebuilds. No `bundle_is_valid` code change needed beyond the constant. No manual migration. The live Hub card updates only on next publish; card-only re-upload via existing `publish_model_card_command`.

## Open Questions

None blocking. Minor open item: exact stable title string — "Awale AlphaZero-like" proposed; tasks may pin the literal.
