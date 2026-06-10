"""Local-only, language-agnostic AI source-code detector.

`SourceCodeAIDetector.validate(source_or_path)` inspects a source string (or a
local file path) and returns a structured verdict describing how likely the code
was produced or contaminated by an AI assistant.

Design rules (see
``com_subjective_architecture_docs/worktwins/01_qa_pytest_contracts_source_code_ai_detector.md``):

* No network access. Never call remote LLMs, APIs, or cloud services.
* Strong, locally verifiable provenance signals are evaluated first and may stop
  the pipeline early ("terminal" detectors).
* Weaker style heuristics only accumulate evidence; they are never terminal.
* Output is deterministic for a given input.
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable

__all__ = ["SourceCodeAIDetector"]


# Each detector returns a dict shaped like:
#   {"id", "confidence", "weight", "terminal", "evidence"}
# or ``None`` when it does not fire.
DetectorResult = dict


class SourceCodeAIDetector:
    """Heuristic, offline detector of AI-generated / AI-contaminated source code."""

    # --- Label / confidence thresholds -----------------------------------
    STRONG_THRESHOLD = 0.90
    LIKELY_THRESHOLD = 0.75
    INCONCLUSIVE_THRESHOLD = 0.45

    # Non-terminal evidence is capped so it can never masquerade as the strong,
    # provenance-backed verdict reserved for terminal detectors.
    NON_TERMINAL_CAP = 0.89

    # Heuristics decide a string is a path only when it is reasonably short and
    # actually exists on disk.
    MAX_PATH_LEN = 4096

    # ------------------------------------------------------------------ API
    @classmethod
    def validate(cls, source_or_path: Any, is_path: bool | None = None) -> dict:
        """Validate a source string or a path to a local source file.

        Returns a dictionary with the public contract keys. Never raises on
        ordinary input; unreadable files produce a safe low-evidence result.
        """
        input_type, source, read_error = cls._load_source(source_or_path, is_path)

        if read_error is not None:
            result = cls._empty_result(input_type)
            result["evidence"] = [read_error]
            return result

        if not source:
            return cls._empty_result(input_type)

        language = cls._detect_language(source_or_path, source, input_type)

        detectors: list[DetectorResult] = []
        evidence: list[str] = []
        stopped_early = False
        stopped_by: str | None = None
        terminal_fired = False
        ai_likelihood = 0.0

        # Detectors are evaluated strictly in priority order. A terminal hit
        # stops evaluation immediately; lower-priority detectors are skipped.
        for detector_id, terminal, fn in cls._detector_pipeline():
            hit = fn(source)
            if not hit:
                continue

            record = {
                "id": detector_id,
                "confidence": round(float(hit["confidence"]), 4),
                "weight": float(hit.get("weight", hit["confidence"])),
                "terminal": bool(terminal),
                "evidence": list(hit["evidence"]),
            }
            detectors.append(record)
            evidence.extend(record["evidence"])

            if terminal:
                ai_likelihood = record["confidence"]
                terminal_fired = True
                stopped_early = True
                stopped_by = detector_id
                break

        if not terminal_fired:
            ai_likelihood = cls._combine_non_terminal(detectors)

        label = cls._label_for(ai_likelihood, terminal_fired)
        confidence = cls._confidence_for(ai_likelihood)

        return {
            "ai_likelihood": round(ai_likelihood, 4),
            "label": label,
            "confidence": confidence,
            "stopped_early": stopped_early,
            "stopped_by": stopped_by,
            "evidence": evidence,
            "detectors": detectors,
            "input_type": input_type,
            "language": language,
        }

    # -------------------------------------------------------- input loading
    @classmethod
    def _load_source(
        cls, source_or_path: Any, is_path: bool | None
    ) -> tuple[str, str, str | None]:
        """Return ``(input_type, source_text, read_error)``.

        ``read_error`` is ``None`` unless a path was given but could not be
        decoded as text.
        """
        looks_like_path = cls._is_existing_path(source_or_path) if is_path is None else is_path

        if looks_like_path:
            try:
                with open(os.fspath(source_or_path), "r", encoding="utf-8") as handle:
                    return "path", handle.read(), None
            except (UnicodeDecodeError, ValueError):
                return "path", "", "Unable to decode source file"
            except OSError:
                # Path was requested but unreadable; degrade to a safe result.
                return "path", "", "Unable to decode source file"

        return "source", "" if source_or_path is None else str(source_or_path), None

    @classmethod
    def _is_existing_path(cls, candidate: Any) -> bool:
        if isinstance(candidate, os.PathLike):
            candidate = os.fspath(candidate)
        if not isinstance(candidate, str):
            return False
        if not candidate or "\n" in candidate or len(candidate) > cls.MAX_PATH_LEN:
            return False
        try:
            return os.path.isfile(candidate)
        except (OSError, ValueError):
            return False

    # ----------------------------------------------------------- detectors
    @classmethod
    def _detector_pipeline(cls) -> list[tuple[str, bool, Callable[[str], DetectorResult | None]]]:
        """Detectors in strict priority order: (id, terminal, function)."""
        return [
            ("explicit_ai_provenance", True, cls._d_explicit_ai_provenance),
            ("ai_chat_artifacts", True, cls._d_ai_chat_artifacts),
            ("markdown_code_fence_remnants", True, cls._d_markdown_code_fence_remnants),
            ("prompt_instruction_remnants", True, cls._d_prompt_instruction_remnants),
            ("generated_response_structure", False, cls._d_generated_response_structure),
            ("placeholder_values", False, cls._d_placeholder_values),
            ("repeated_boilerplate_chunks", False, cls._d_repeated_boilerplate_chunks),
            ("tutorial_comments", False, cls._d_tutorial_comments),
            ("generic_naming", False, cls._d_generic_naming),
            ("low_else_ratio", False, cls._d_low_else_ratio),
        ]

    # 1) explicit_ai_provenance --------------------------------------------
    _PROVENANCE_PATTERNS = [
        r"generated by (?:chatgpt|gpt-?\d?|gpt|claude|copilot|gemini|bard|an ai|an? ai)",
        r"written by (?:chatgpt|gpt-?\d?|claude|copilot|an? ai)",
        r"created (?:by|with) (?:chatgpt|gpt-?\d?|claude|copilot|gemini)",
        r"this code was generated by",
        r"as an ai language model",
        r"auto-?generated by (?:chatgpt|gpt|claude|copilot)",
    ]

    @classmethod
    def _d_explicit_ai_provenance(cls, source: str) -> DetectorResult | None:
        matches = cls._find_matches(source, cls._PROVENANCE_PATTERNS)
        if not matches:
            return None
        return {"confidence": 0.97, "weight": 1.0, "evidence": matches}

    # 2) ai_chat_artifacts --------------------------------------------------
    _CHAT_PATTERNS = [
        r"sure,? here(?:'s| is)\b[^\n]*",
        r"here(?:'s| is) (?:the|a|an|your) (?:implementation|code|function|solution|example)[^\n]*",
        r"certainly[!,][^\n]*",
        r"of course[!,][^\n]*",
        r"i hope this helps[^\n]*",
        r"let me know if (?:you|there|this)[^\n]*",
        r"feel free to (?:ask|modify|adjust)[^\n]*",
        r"as requested[, ][^\n]*",
    ]

    @classmethod
    def _d_ai_chat_artifacts(cls, source: str) -> DetectorResult | None:
        matches = cls._find_matches(source, cls._CHAT_PATTERNS)
        if not matches:
            return None
        evidence = [f"AI chat artifact: {m.strip()}" for m in matches]
        return {"confidence": 0.88, "weight": 0.95, "evidence": evidence}

    # 3) markdown_code_fence_remnants --------------------------------------
    @classmethod
    def _d_markdown_code_fence_remnants(cls, source: str) -> DetectorResult | None:
        fences = re.findall(r"^[ \t]*```[a-zA-Z0-9_+-]*[ \t]*$", source, re.MULTILINE)
        if not fences:
            return None
        evidence = [f"Markdown code fence remnant: {f.strip()}" for f in fences]
        return {"confidence": 0.80, "weight": 0.9, "evidence": evidence}

    # 4) prompt_instruction_remnants ---------------------------------------
    _PROMPT_PATTERNS = [
        r"implement the following[^\n]*",
        r"return only the code[^\n]*",
        r"write a (?:function|program|method|class|script) (?:that|to)[^\n]*",
        r"complete the following[^\n]*",
        r"your task is to[^\n]*",
        r"do not include (?:any )?explanations?[^\n]*",
        r"please (?:implement|write|complete|generate)[^\n]*",
    ]

    @classmethod
    def _d_prompt_instruction_remnants(cls, source: str) -> DetectorResult | None:
        matches = cls._find_matches(source, cls._PROMPT_PATTERNS)
        if not matches:
            return None
        evidence = [f"Prompt instruction remnant: {m.strip()}" for m in matches]
        return {"confidence": 0.76, "weight": 0.85, "evidence": evidence}

    # 5) generated_response_structure --------------------------------------
    _STRUCTURE_PATTERNS = [
        r"^[ \t]*#+\s*(?:explanation|overview|usage|example|step-by-step|how it works)\b[^\n]*",
        r"^[ \t]*#\s*step\s*\d+[:.][^\n]*",
        r"^[ \t]*#\s*\d+[.)]\s+\w[^\n]*",
        r"here(?:'s| is) how (?:it works|to use)[^\n]*",
    ]

    @classmethod
    def _d_generated_response_structure(cls, source: str) -> DetectorResult | None:
        matches = []
        for pattern in cls._STRUCTURE_PATTERNS:
            for m in re.findall(pattern, source, re.IGNORECASE | re.MULTILINE):
                matches.append(m.strip())
        if not matches:
            return None
        evidence = [f"Generated-response structure: {m}" for m in cls._dedup(matches)]
        return {"confidence": 0.35, "weight": 0.5, "evidence": evidence}

    # 6) placeholder_values -------------------------------------------------
    _PLACEHOLDER_PATTERNS = [
        r"your[-_](?:api[-_]?key|api|secret|token|password|username)",
        r"<[^>\n]*(?:your|api[_-]?key|token|secret|placeholder)[^>\n]*>",
        r"https?://(?:www\.)?example\.(?:com|org|net)\S*",
        r"\b(?:insert|replace|change)[-_](?:me|this|here|your)[-_a-z]*\b",
        r"(?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"](?:xxx+|changeme|todo|placeholder|your[-_][a-z]+)['\"]",
        r"\bTODO[: ]\s*(?:replace|set|add|fill)\b",
    ]

    @classmethod
    def _d_placeholder_values(cls, source: str) -> DetectorResult | None:
        matches = cls._find_matches(source, cls._PLACEHOLDER_PATTERNS)
        if not matches:
            return None
        evidence = [f"Placeholder value: {m.strip()}" for m in cls._dedup(matches)]
        return {"confidence": 0.45, "weight": 0.6, "evidence": evidence}

    # 7) repeated_boilerplate_chunks ---------------------------------------
    @classmethod
    def _d_repeated_boilerplate_chunks(cls, source: str) -> DetectorResult | None:
        lines = [ln.strip() for ln in source.splitlines()]
        # Only consider substantive lines (ignore short / import / brace lines).
        meaningful = [
            ln
            for ln in lines
            if len(ln) >= 20
            and not ln.startswith(("import ", "from "))
            and ln not in {"{", "}", "()", "):"}
        ]
        counts: dict[str, int] = {}
        for ln in meaningful:
            counts[ln] = counts.get(ln, 0) + 1
        repeated = [ln for ln, c in counts.items() if c >= 3]
        if not repeated:
            return None
        evidence = [f"Repeated boilerplate block (x{counts[ln]}): {ln[:60]}" for ln in repeated]
        return {"confidence": 0.30, "weight": 0.45, "evidence": evidence}

    # 8) tutorial_comments --------------------------------------------------
    _TUTORIAL_PATTERNS = [
        r"^[ \t]*(?://|#)\s*(?:loop through|iterate over|iterate through)\b[^\n]*",
        r"^[ \t]*(?://|#)\s*return the (?:result|value|output|answer)\b[^\n]*",
        r"^[ \t]*(?://|#)\s*(?:check if|create a|create an|initialize|define the|set the|get the|print the|add the|store the)\b[^\n]*",
        r"^[ \t]*(?://|#)\s*(?:now we|first,? we|next,? we|then we|finally,? we)\b[^\n]*",
    ]

    @classmethod
    def _d_tutorial_comments(cls, source: str) -> DetectorResult | None:
        matches = []
        for pattern in cls._TUTORIAL_PATTERNS:
            for m in re.findall(pattern, source, re.IGNORECASE | re.MULTILINE):
                matches.append(m.strip())
        if not matches:
            return None
        evidence = [f"Tutorial comment: {m}" for m in cls._dedup(matches)]
        return {"confidence": 0.30, "weight": 0.4, "evidence": evidence}

    # 9) generic_naming -----------------------------------------------------
    _GENERIC_NAMES = {
        "data", "result", "results", "temp", "tmp", "foo", "bar", "baz",
        "value", "val", "item", "items", "obj", "res", "arr", "elem",
    }

    @classmethod
    def _d_generic_naming(cls, source: str) -> DetectorResult | None:
        assigned = re.findall(r"^[ \t]*([A-Za-z_]\w*)\s*=", source, re.MULTILINE)
        params = re.findall(r"\bdef\s+\w+\s*\(([^)]*)\)", source)
        names = [n.lower() for n in assigned]
        for group in params:
            for token in group.split(","):
                token = token.strip().split(":")[0].split("=")[0].strip()
                if token:
                    names.append(token.lower())
        generic_hits = sorted({n for n in names if n in cls._GENERIC_NAMES})
        # Require several distinct generic identifiers to avoid flagging the
        # incidental ``result`` in clean human code.
        if len(generic_hits) < 4:
            return None
        evidence = [f"Generic identifier naming: {', '.join(generic_hits)}"]
        return {"confidence": 0.20, "weight": 0.35, "evidence": evidence}

    # 10) low_else_ratio ----------------------------------------------------
    @classmethod
    def _d_low_else_ratio(cls, source: str) -> DetectorResult | None:
        if_count = len(re.findall(r"^[ \t]*(?:el)?if\b", source, re.MULTILINE))
        else_count = len(re.findall(r"^[ \t]*else\b", source, re.MULTILINE))
        if if_count >= 4 and else_count == 0:
            evidence = [f"Low else ratio: {if_count} if-branches, 0 else-branches"]
            return {"confidence": 0.20, "weight": 0.3, "evidence": evidence}
        return None

    # --------------------------------------------------------- scoring math
    @classmethod
    def _combine_non_terminal(cls, detectors: list[DetectorResult]) -> float:
        """Noisy-OR accumulation of non-terminal evidence, capped below strong."""
        product = 1.0
        for record in detectors:
            product *= 1.0 - max(0.0, min(1.0, record["confidence"]))
        likelihood = 1.0 - product
        return min(likelihood, cls.NON_TERMINAL_CAP)

    @classmethod
    def _label_for(cls, ai_likelihood: float, terminal_fired: bool) -> str:
        if ai_likelihood >= cls.STRONG_THRESHOLD and terminal_fired:
            return "strong_ai_evidence"
        if ai_likelihood >= cls.LIKELY_THRESHOLD:
            return "likely_ai_assisted"
        if ai_likelihood >= cls.INCONCLUSIVE_THRESHOLD:
            return "inconclusive"
        return "low_ai_evidence"

    @classmethod
    def _confidence_for(cls, ai_likelihood: float) -> str:
        if ai_likelihood >= 0.85:
            return "high"
        if ai_likelihood >= 0.55:
            return "medium"
        return "low"

    # ----------------------------------------------------------- utilities
    @classmethod
    def _find_matches(cls, source: str, patterns: list[str]) -> list[str]:
        found: list[str] = []
        for pattern in patterns:
            for m in re.finditer(pattern, source, re.IGNORECASE):
                found.append(m.group(0))
        return cls._dedup(found)

    @staticmethod
    def _dedup(items: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            key = item.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(item)
        return ordered

    @staticmethod
    def _detect_language(source_or_path: Any, source: str, input_type: str) -> str | None:
        ext_map = {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".java": "java", ".go": "go", ".rb": "ruby", ".rs": "rust",
            ".c": "c", ".cpp": "cpp", ".cc": "cpp", ".cs": "csharp",
            ".php": "php", ".swift": "swift", ".kt": "kotlin", ".sh": "bash",
        }
        if input_type == "path":
            try:
                _, ext = os.path.splitext(os.fspath(source_or_path))
                if ext.lower() in ext_map:
                    return ext_map[ext.lower()]
            except (TypeError, ValueError):
                pass
        # Content heuristics for raw source strings.
        if re.search(r"^\s*(?:def |class |import |from \w+ import )", source, re.MULTILINE):
            return "python"
        if re.search(r"\b(?:function|const|let|var)\b.*[;{]", source):
            return "javascript"
        return None

    @classmethod
    def _empty_result(cls, input_type: str) -> dict:
        return {
            "ai_likelihood": 0.0,
            "label": "low_ai_evidence",
            "confidence": "low",
            "stopped_early": False,
            "stopped_by": None,
            "evidence": [],
            "detectors": [],
            "input_type": input_type,
            "language": None,
        }
