# Test Fixtures — Attribution & Policy

All fixtures here are loaded from disk only; **no network access** occurs during
`pytest`. They are reproducible offline.

## Directory layout

| Folder | Purpose |
|--------|---------|
| `ai_artifacts/` | Synthetic samples containing explicit AI provenance / chat / fence / prompt contamination. Pattern tests, not copyright-dependent. |
| `human_samples/` | Clean, human-style negative controls (no contamination signals). |
| `mixed_samples/` | Samples that accumulate several medium signals without any terminal signal. |

## Attribution

The committed tests must be reproducible offline, so GitHub fetching at test time
is forbidden. Per the QA contract:

- **`ai_artifacts/`** — **synthetic.** AI-contaminated examples are pattern tests
  (they assert on provenance/chat/fence/prompt markers), not copyrighted code, so
  synthetic fixtures are acceptable and preferred.
- **`mixed_samples/`** — **synthetic.** Hand-authored to combine placeholder values,
  tutorial comments, a generated-response header, and repeated boilerplate.
- **`human_samples/`** — **hand-authored clean code** used as negative controls.
  These are small, well-known algorithm implementations (quicksort, an LRU cache)
  written from scratch for this suite. They are intentionally free of AI tells.

> Note on the GitHub-snippet policy: the QA prompt allows vendoring small,
> permissively licensed human snippets from GitHub *with attribution*. To keep the
> suite fully offline and free of licensing ambiguity, the human negative controls
> here are original hand-authored implementations rather than vendored GitHub code.
> If GitHub-derived snippets are later added, each must record: source repository
> URL, license name, file path, and retrieval date in this file.

| File | Origin | License | Retrieved |
|------|--------|---------|-----------|
| `human_samples/quicksort.py` | Original (hand-authored) | N/A | — |
| `human_samples/lru_cache.py` | Original (hand-authored) | N/A | — |
| `mixed_samples/contaminated_config.py` | Synthetic | N/A | — |
| `ai_artifacts/chatgpt_generated.py` | Synthetic | N/A | — |
