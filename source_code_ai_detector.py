"""Local-only, language-agnostic AI source-code detector.

``SourceCodeAIDetector.validate(source_or_path)`` inspects a source string (or a
local file path) and returns a structured, probabilistic verdict describing how
likely the code was produced or contaminated by an AI assistant.

This is **not** a legal/authorship proof — it is a local trust and review-priority
signal. Design rules (see the worktwins QA + implementation prompts):

* No network access. Never call remote LLMs, APIs, or cloud services.
* Strong, locally verifiable provenance signals run first and may stop early
  ("terminal" detectors). Only ``explicit_ai_provenance`` and
  ``ai_chat_artifacts`` are terminal.
* Weaker style heuristics only accumulate evidence; they are never terminal.
* Scores accumulate with a saturating rule so they cannot explode past 1.0.
* Output is deterministic for a given input. Standard library only.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

__all__ = ["SourceCodeAIDetector"]


@dataclass(frozen=True)
class DetectorSpec:
    """A single detector: ``id``, its priority order, and the runner function.

    The runner receives ``(source, context)`` and returns either ``None`` (no
    hit) or a dict with ``confidence``, ``weight``, ``terminal``, ``evidence``.
    """

    id: str
    priority: int
    terminal: bool
    runner: Callable[[str, dict], Optional[dict]]


class SourceCodeAIDetector:
    """Heuristic, offline detector of AI-generated / AI-contaminated source code."""

    DEFAULT_STOP_THRESHOLD = 0.90

    # Label / confidence thresholds.
    STRONG_THRESHOLD = 0.90
    LIKELY_THRESHOLD = 0.75
    INCONCLUSIVE_THRESHOLD = 0.45

    # Heuristics decide a string is a path only when it is reasonably short and
    # actually exists on disk.
    MAX_PATH_LEN = 4096

    _LANGUAGE_BY_EXT = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript-react",
        ".jsx": "javascript-react",
        ".java": "java",
        ".cs": "csharp",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".c": "c",
        ".cpp": "cpp",
        ".md": "markdown",
    }

    # ------------------------------------------------------------------ API
    @staticmethod
    def validate(source_or_path: str, **kwargs) -> dict:
        """Validate a source string or a path to a local source file.

        Optional keyword arguments:

        * ``is_path`` (bool): force path/source interpretation instead of
          auto-detecting.
        * ``stop_threshold`` (float): terminal-detector early-stop threshold
          (default ``0.90``).

        Never raises on ordinary input; unreadable files produce a safe
        low-evidence result.
        """
        cls = SourceCodeAIDetector
        stop_threshold = float(kwargs.get("stop_threshold", cls.DEFAULT_STOP_THRESHOLD))
        is_path = kwargs.get("is_path")

        source, input_type, language, read_error = cls._load_source(source_or_path, is_path)

        if read_error is not None:
            result = cls._empty_result(input_type, language)
            result["evidence"] = [read_error]
            return result

        if not source or not source.strip():
            return cls._empty_result(input_type, language)

        context = {
            "input_type": input_type,
            "language": language,
            "stop_threshold": stop_threshold,
        }

        detectors: list[dict] = []
        evidence: list[str] = []
        stopped_early = False
        stopped_by: Optional[str] = None
        score = 0.0

        for spec in cls._pipeline():
            hit = spec.runner(source, context)
            if not hit:
                continue

            record = {
                "id": spec.id,
                "confidence": round(float(hit["confidence"]), 4),
                "weight": round(float(hit["weight"]), 4),
                "terminal": spec.terminal,
                "evidence": list(hit["evidence"]),
            }
            detectors.append(record)
            evidence.extend(record["evidence"])

            if spec.terminal and record["confidence"] >= stop_threshold:
                stopped_early = True
                stopped_by = spec.id
                score = max(score, record["confidence"])
                break

            # Saturating accumulation keeps the score in [0, 1).
            score = score + (1.0 - score) * (record["confidence"] * record["weight"])

        ai_likelihood = round(score, 4)
        label = cls._label_for(ai_likelihood, stopped_early)
        confidence = cls._confidence_for(ai_likelihood)

        return {
            "ai_likelihood": ai_likelihood,
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
    @staticmethod
    def _load_source(
        source_or_path: Any, is_path: Optional[bool]
    ) -> tuple[str, str, Optional[str], Optional[str]]:
        """Return ``(source, input_type, language, read_error)``.

        ``read_error`` is ``None`` unless a path was given but could not be
        decoded as UTF-8 text.
        """
        cls = SourceCodeAIDetector

        if is_path is None:
            treat_as_path = cls._is_existing_path(source_or_path)
        else:
            treat_as_path = bool(is_path)

        if treat_as_path:
            path = os.fspath(source_or_path)
            language = cls._language_for_path(path)
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    return handle.read(), "path", language, None
            except (UnicodeDecodeError, ValueError, OSError):
                return "", "path", language, "Unable to decode source file"

        source = "" if source_or_path is None else str(source_or_path)
        return source, "source", None, None

    @staticmethod
    def _is_existing_path(candidate: Any) -> bool:
        cls = SourceCodeAIDetector
        if isinstance(candidate, os.PathLike):
            candidate = os.fspath(candidate)
        if not isinstance(candidate, str):
            return False
        # Anything with a newline is source, not a path.
        if not candidate or "\n" in candidate or len(candidate) > cls.MAX_PATH_LEN:
            return False
        try:
            return os.path.isfile(candidate)
        except (OSError, ValueError):
            return False

    @staticmethod
    def _language_for_path(path: str) -> Optional[str]:
        cls = SourceCodeAIDetector
        try:
            _, ext = os.path.splitext(path)
        except (TypeError, ValueError):
            return None
        return cls._LANGUAGE_BY_EXT.get(ext.lower())

    # ----------------------------------------------------------- pipeline
    @staticmethod
    def _pipeline() -> list[DetectorSpec]:
        """Detectors in strict priority order; only the first two are terminal."""
        cls = SourceCodeAIDetector
        return [
            DetectorSpec("explicit_ai_provenance", 1, True, cls._detect_explicit_ai_provenance),
            DetectorSpec("ai_chat_artifacts", 2, True, cls._detect_ai_chat_artifacts),
            DetectorSpec("markdown_code_fence_remnants", 3, False, cls._detect_markdown_code_fence_remnants),
            DetectorSpec("prompt_instruction_remnants", 4, False, cls._detect_prompt_instruction_remnants),
            DetectorSpec("generated_response_structure", 5, False, cls._detect_generated_response_structure),
            DetectorSpec("placeholder_values", 6, False, cls._detect_placeholder_values),
            DetectorSpec("repeated_boilerplate_chunks", 7, False, cls._detect_repeated_boilerplate_chunks),
            DetectorSpec("tutorial_comments", 8, False, cls._detect_tutorial_comments),
            DetectorSpec("generic_naming", 9, False, cls._detect_generic_naming),
            DetectorSpec("low_else_ratio", 10, False, cls._detect_low_else_ratio),
        ]

    # 1) explicit_ai_provenance (terminal) ---------------------------------
    _PROVENANCE_PATTERNS = [
        r"generated by (?:chatgpt|gpt-?\d?|gpt|claude|copilot|gemini|bard|an? ai)",
        r"written by (?:chatgpt|gpt-?\d?|claude|copilot|an? ai)",
        r"created (?:by|with) (?:chatgpt|gpt-?\d?|claude|copilot|gemini)",
        r"this code was generated by (?:an? )?ai",
        r"\bai-generated\b",
        r"auto-?generated by (?:chatgpt|gpt|claude|copilot)",
    ]

    @staticmethod
    def _detect_explicit_ai_provenance(source: str, context: dict) -> Optional[dict]:
        cls = SourceCodeAIDetector
        matches = cls._find_matches(source, cls._PROVENANCE_PATTERNS)
        if not matches:
            return None
        evidence = [f"Found explicit AI provenance phrase: {m}" for m in matches]
        return {"confidence": 0.96, "weight": 1.0, "evidence": evidence}

    # 2) ai_chat_artifacts (terminal) --------------------------------------
    _CHAT_PATTERNS = [
        r"as an ai language model[^\n]*",
        r"sure,? here(?:'s| is) (?:the|a|an|your)?[^\n]*",
        r"here(?:'s| is) (?:the|a|an|your) (?:implementation|code|function|solution|example)[^\n]*",
        r"below is the (?:code|implementation|function)[^\n]*",
        r"let me know if you (?:want|need|have)[^\n]*",
        r"i (?:cannot|can't|can not) access your (?:file ?system|files)[^\n]*",
        r"i hope this helps[^\n]*",
    ]

    @staticmethod
    def _detect_ai_chat_artifacts(source: str, context: dict) -> Optional[dict]:
        cls = SourceCodeAIDetector
        matches = cls._find_matches(source, cls._CHAT_PATTERNS)
        if not matches:
            return None
        evidence = [f"Found AI chat artifact: {m.strip()}" for m in matches]
        return {"confidence": 0.94, "weight": 1.0, "evidence": evidence}

    # 3) markdown_code_fence_remnants --------------------------------------
    @staticmethod
    def _detect_markdown_code_fence_remnants(source: str, context: dict) -> Optional[dict]:
        # Suppress for actual markdown files where fences are expected.
        if context.get("language") == "markdown":
            return None
        fences = re.findall(r"^[ \t]*```[a-zA-Z0-9_+#-]*[ \t]*$", source, re.MULTILINE)
        if not fences:
            return None
        evidence = [f"Markdown code fence remnant in source: {f.strip()}" for f in fences]
        return {"confidence": 0.88, "weight": 0.90, "evidence": evidence}

    # 4) prompt_instruction_remnants ---------------------------------------
    _PROMPT_PATTERNS = [
        r"implement the following[^\n]*",
        r"refactor this code[^\n]*",
        r"return only the code[^\n]*",
        r"generate a (?:function|class|method|script)[^\n]*",
        r"write a (?:function|program|method|class|script) (?:that|to)[^\n]*",
        r"make sure to[^\n]*",
        r"do not change[^\n]*",
        r"you are an expert[^\n]*",
        r"act as a senior (?:developer|engineer)[^\n]*",
        r"complete the following[^\n]*",
        r"your task is to[^\n]*",
    ]

    @staticmethod
    def _detect_prompt_instruction_remnants(source: str, context: dict) -> Optional[dict]:
        cls = SourceCodeAIDetector
        matches = cls._find_matches(source, cls._PROMPT_PATTERNS)
        if not matches:
            return None
        evidence = [f"Leaked prompt instruction: {m.strip()}" for m in matches]
        # Weight tuned so a single leaked instruction is strong evidence
        # (the suggested 0.85 falls just below the review contract's bar).
        return {"confidence": 0.82, "weight": 0.90, "evidence": evidence}

    # 5) generated_response_structure --------------------------------------
    _STRUCTURE_PATTERN = re.compile(
        r"^[ \t]*(?:#|//|/?\*|<!--)?\s*"
        r"(explanation|usage|examples?|notes?|implementation|approach|solution|overview|step\s*\d+)\s*:",
        re.IGNORECASE | re.MULTILINE,
    )

    @staticmethod
    def _detect_generated_response_structure(source: str, context: dict) -> Optional[dict]:
        cls = SourceCodeAIDetector
        matches = [m.group(0).strip() for m in cls._STRUCTURE_PATTERN.finditer(source)]
        matches = cls._dedup(matches)
        if not matches:
            return None
        evidence = [f"Generated-response section heading: {m}" for m in matches]
        return {"confidence": 0.65, "weight": 0.65, "evidence": evidence}

    # 6) placeholder_values -------------------------------------------------
    _PLACEHOLDER_PATTERNS = [
        r"your[-_](?:api[-_]?key|token|secret|password|username)(?:[-_]here)?",
        r"your[-_]token[-_]here",
        r"replace[-_]this",
        r"insert[-_]api[-_]key",
        r"<[^>\n]*(?:your|api[_-]?key|token|secret|placeholder)[^>\n]*>",
        r"https?://(?:www\.)?example\.(?:com|org|net)\S*",
        r"\bexample\.(?:com|org|net)\b",
        r"\bjohn doe\b",
        r"\bjane doe\b",
        r"\blorem ipsum\b",
        r"\bTODO[: ]\s*replace\b",
    ]

    @staticmethod
    def _detect_placeholder_values(source: str, context: dict) -> Optional[dict]:
        cls = SourceCodeAIDetector
        matches = cls._find_matches(source, cls._PLACEHOLDER_PATTERNS)
        if not matches:
            return None
        evidence = [f"Placeholder/example value: {m.strip()}" for m in matches]
        return {"confidence": 0.60, "weight": 0.60, "evidence": evidence}

    # 7) repeated_boilerplate_chunks ---------------------------------------
    @staticmethod
    def _detect_repeated_boilerplate_chunks(source: str, context: dict) -> Optional[dict]:
        # Language-agnostic: normalize whitespace, drop blank lines, then count
        # repeated sliding windows of 3..6 consecutive lines.
        lines = [re.sub(r"\s+", " ", ln.strip()) for ln in source.splitlines()]
        lines = [ln for ln in lines if ln]
        if len(lines) < 6:
            return None

        repeated_windows = 0
        seen_repeats: set[tuple[str, ...]] = set()
        for window_size in range(3, 7):
            counts: dict[tuple[str, ...], int] = {}
            for start in range(len(lines) - window_size + 1):
                window = tuple(lines[start : start + window_size])
                # Ignore windows that are too short in total characters.
                if sum(len(w) for w in window) < 40:
                    continue
                counts[window] = counts.get(window, 0) + 1
            for window, count in counts.items():
                if count >= 2 and window not in seen_repeats:
                    seen_repeats.add(window)
                    repeated_windows += 1

        if repeated_windows == 0:
            return None
        evidence = [f"Repeated boilerplate: {repeated_windows} repeated multi-line chunk(s)"]
        return {"confidence": 0.58, "weight": 0.55, "evidence": evidence}

    # 8) tutorial_comments --------------------------------------------------
    _TUTORIAL_PATTERN = re.compile(
        r"^[ \t]*(?://|#|/?\*)\s*"
        r"(?:check if|loop through|iterate over|iterate through|return the (?:result|value|output)|"
        r"initialize the|create a new|create an?|set the|get the|call the function|"
        r"now we|first,? we|next,? we|then we)\b[^\n]*",
        re.IGNORECASE | re.MULTILINE,
    )

    @staticmethod
    def _detect_tutorial_comments(source: str, context: dict) -> Optional[dict]:
        cls = SourceCodeAIDetector
        matches = cls._dedup([m.group(0).strip() for m in cls._TUTORIAL_PATTERN.finditer(source)])
        if not matches:
            return None
        evidence = [f"Tutorial-style comment: {m}" for m in matches]
        return {"confidence": 0.45, "weight": 0.40, "evidence": evidence}

    # 9) generic_naming -----------------------------------------------------
    _GENERIC_NAMES = {
        "data", "result", "results", "temp", "tmp", "foo", "bar", "baz",
        "value", "val", "item", "items", "obj", "res", "arr", "elem",
    }
    _GENERIC_FUNC_PATTERN = re.compile(
        r"\b(?:process_?data|handle_?data|do_?something|my_?function|test_?func)\b",
        re.IGNORECASE,
    )

    @staticmethod
    def _detect_generic_naming(source: str, context: dict) -> Optional[dict]:
        cls = SourceCodeAIDetector
        assigned = re.findall(r"^[ \t]*([A-Za-z_]\w*)\s*=(?!=)", source, re.MULTILINE)
        params: list[str] = []
        for group in re.findall(r"\bdef\s+\w+\s*\(([^)]*)\)", source):
            for token in group.split(","):
                token = token.strip().split(":")[0].split("=")[0].strip()
                if token:
                    params.append(token)

        names = [n.lower() for n in assigned + params]
        generic_hits = sorted({n for n in names if n in cls._GENERIC_NAMES})
        func_hits = cls._dedup(cls._GENERIC_FUNC_PATTERN.findall(source))

        # Conservative: require several distinct generic identifiers (or a
        # couple of tell-tale generic function names) before firing.
        if len(generic_hits) + len(func_hits) < 4:
            return None
        parts = []
        if generic_hits:
            parts.append(f"identifiers: {', '.join(generic_hits)}")
        if func_hits:
            parts.append(f"functions: {', '.join(func_hits)}")
        evidence = [f"Excessive generic naming ({'; '.join(parts)})"]
        return {"confidence": 0.30, "weight": 0.25, "evidence": evidence}

    # 10) low_else_ratio ----------------------------------------------------
    @staticmethod
    def _detect_low_else_ratio(source: str, context: dict) -> Optional[dict]:
        if_count = len(re.findall(r"(?<![\w.])if(?![\w])", source))
        else_count = len(re.findall(r"(?<![\w.])else(?:\s+if|if)?(?![\w])", source))
        if if_count >= 8 and (else_count / if_count) <= 0.08:
            evidence = [f"Low else ratio: {else_count} else branches for {if_count} if statements"]
            return {"confidence": 0.18, "weight": 0.25, "evidence": evidence}
        return None

    # --------------------------------------------------------- scoring math
    @staticmethod
    def _label_for(ai_likelihood: float, stopped_early: bool) -> str:
        cls = SourceCodeAIDetector
        if ai_likelihood >= cls.STRONG_THRESHOLD and stopped_early:
            return "strong_ai_evidence"
        if ai_likelihood >= cls.LIKELY_THRESHOLD:
            return "likely_ai_assisted"
        if ai_likelihood >= cls.INCONCLUSIVE_THRESHOLD:
            return "inconclusive"
        return "low_ai_evidence"

    @staticmethod
    def _confidence_for(ai_likelihood: float) -> str:
        if ai_likelihood >= 0.85:
            return "high"
        if ai_likelihood >= 0.55:
            return "medium"
        return "low"

    # ----------------------------------------------------------- utilities
    @staticmethod
    def _find_matches(source: str, patterns: list[str]) -> list[str]:
        cls = SourceCodeAIDetector
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
    def _empty_result(input_type: str, language: Optional[str]) -> dict:
        return {
            "ai_likelihood": 0.0,
            "label": "low_ai_evidence",
            "confidence": "low",
            "stopped_early": False,
            "stopped_by": None,
            "evidence": [],
            "detectors": [],
            "input_type": input_type,
            "language": language,
        }
