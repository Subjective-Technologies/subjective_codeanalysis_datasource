# subjective_codeanalysis_datasource

A Subjective **v2 datasource** that performs **local-only, language-agnostic
AI source-code detection**. Given a source string or a path to a local file, it
returns a structured verdict describing how likely the code was produced or
contaminated by an AI assistant.

> **No network access.** The detector never calls remote LLMs, remote APIs,
> internet services, or cloud services during validation.

## Components

| File | Role |
|------|------|
| `source_code_ai_detector.py` | Pure, offline `SourceCodeAIDetector.validate()` engine. |
| `SubjectiveCodeanalysisDataSource.py` | v2 datasource wrapper (connection/request/output schemas, `run()`, chat). |
| `tests/` | Pytest **contract suite** (written before/alongside the engine) + offline fixtures. |

## Detection engine

```python
from source_code_ai_detector import SourceCodeAIDetector

result = SourceCodeAIDetector.validate(source_or_path)
```

`source_or_path` may be a raw source string or a path to a local file; the
detector auto-detects an existing file path. The result dict:

```python
{
    "ai_likelihood": float,   # 0.0 .. 1.0
    "label": str,             # strong_ai_evidence | likely_ai_assisted | inconclusive | low_ai_evidence
    "confidence": str,        # high | medium | low
    "stopped_early": bool,
    "stopped_by": str | None,
    "evidence": list[str],
    "detectors": list[dict],  # {id, confidence, weight, terminal, evidence}
    "input_type": str,        # source | path
    "language": str | None,
}
```

### Detector priority

Strong, locally verifiable signals are evaluated first and may stop early
(terminal). Weaker style heuristics only accumulate evidence (never terminal):

1. `explicit_ai_provenance` *(terminal)*
2. `ai_chat_artifacts` *(terminal)*
3. `markdown_code_fence_remnants` *(terminal)*
4. `prompt_instruction_remnants` *(terminal)*
5. `generated_response_structure`
6. `placeholder_values`
7. `repeated_boilerplate_chunks`
8. `tutorial_comments`
9. `generic_naming`
10. `low_else_ratio`

Non-terminal detectors combine via noisy-OR and are capped below the
provenance-backed "strong" verdict.

## Datasource usage

`run(request)` accepts `{"source": "...", "path": "..."}` (path takes
precedence) and returns `ai_likelihood`, `label`, `confidence`, `flagged`
(based on the connection `flag_threshold`), `evidence`, and a JSON `report`.

Chat mode is supported: paste code (or attach files) and the datasource
analyzes each.

## Tests

```bash
pytest -q
```

The contract suite runs fully offline. See `tests/fixtures/README.md` for
fixture attribution.
