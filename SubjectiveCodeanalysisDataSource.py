import os
from typing import Any

from subjective_abstract_data_source_package import SubjectiveDataSource
from brainboost_data_source_logger_package.BBLogger import BBLogger

from source_code_ai_detector import SourceCodeAIDetector


class SubjectiveCodeanalysisDataSource(SubjectiveDataSource):
    """Local, offline source-code analysis datasource.

    Wraps :class:`SourceCodeAIDetector` to classify whether a source string or a
    local file was produced (or contaminated) by an AI assistant. No remote
    LLMs, APIs, or cloud services are contacted.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        conn = getattr(self, "_connection", {}) or {}
        # Likelihood at/above which the result is flagged as AI evidence.
        try:
            self.flag_threshold = float(conn.get("flag_threshold", 0.75))
        except (TypeError, ValueError):
            self.flag_threshold = 0.75

    # ----------------------------------------------------------- schemas
    @classmethod
    def connection_schema(cls) -> dict:
        return {
            "flag_threshold": {
                "type": "number",
                "label": "Flag Threshold",
                "description": "ai_likelihood at/above which 'flagged' is True.",
                "default": 0.75,
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
            },
        }

    @classmethod
    def request_schema(cls) -> dict:
        return {
            "source": {
                "type": "textarea",
                "label": "Source Code",
                "description": "Raw source code to analyze. Ignored if a path is given.",
                "required": False,
                "rows": 12,
            },
            "path": {
                "type": "file_path",
                "label": "Source File Path",
                "description": "Local file to analyze. Takes precedence over 'source'.",
                "required": False,
            },
        }

    @classmethod
    def output_schema(cls) -> dict:
        return {
            "ai_likelihood": {"type": "number", "label": "AI Likelihood (0-1)"},
            "label": {"type": "text", "label": "Verdict Label"},
            "confidence": {"type": "text", "label": "Confidence"},
            "flagged": {"type": "bool", "label": "Flagged as AI"},
            "stopped_by": {"type": "text", "label": "Terminal Detector"},
            "language": {"type": "text", "label": "Detected Language"},
            "input_type": {"type": "text", "label": "Input Type"},
            "evidence": {"type": "textarea", "label": "Evidence"},
            "report": {"type": "textarea", "label": "Full Report (JSON)"},
        }

    @classmethod
    def icon(cls) -> str:
        icon_path = os.path.join(os.path.dirname(__file__), "icon.svg")
        try:
            with open(icon_path, "r", encoding="utf-8") as handle:
                return handle.read()
        except Exception as exc:  # pragma: no cover - icon read is best-effort
            BBLogger.log(f"Error reading icon file: {exc}")
            return ""

    # ----------------------------------------------------------- execution
    def run(self, request: dict) -> Any:
        request = request or {}
        path = request.get("path")
        source = request.get("source")

        if path:
            target: Any = path
            is_path: bool | None = True
        else:
            target = source or ""
            is_path = False

        try:
            report = SourceCodeAIDetector.validate(target, is_path=is_path)
        except Exception as exc:  # defensive: keep the pipeline running
            BBLogger.log(f"Code analysis failed: {exc}")
            return {
                "ai_likelihood": 0.0,
                "label": "low_ai_evidence",
                "confidence": "low",
                "flagged": False,
                "stopped_by": None,
                "language": None,
                "input_type": "source",
                "evidence": "",
                "report": "{}",
                "error": str(exc),
            }

        return self._to_output(report)

    def _to_output(self, report: dict) -> dict:
        import json

        return {
            "ai_likelihood": report["ai_likelihood"],
            "label": report["label"],
            "confidence": report["confidence"],
            "flagged": report["ai_likelihood"] >= self.flag_threshold,
            "stopped_by": report["stopped_by"],
            "language": report["language"],
            "input_type": report["input_type"],
            "evidence": "\n".join(report["evidence"]),
            "report": json.dumps(report, ensure_ascii=False),
        }

    # ----------------------------------------------------------- chat mode
    def supports_chat(self) -> bool:
        return True

    def handle_message(self, message: str, files: list | None = None) -> Any:
        """Analyze code pasted into chat, or files attached to the message."""
        if files:
            outputs = []
            for f in files:
                text = f.get("text")
                if text is None and f.get("path"):
                    report = SourceCodeAIDetector.validate(f["path"], is_path=True)
                else:
                    report = SourceCodeAIDetector.validate(text or "", is_path=False)
                out = self._to_output(report)
                out["name"] = f.get("name")
                outputs.append(out)
            return {"results": outputs}

        report = SourceCodeAIDetector.validate(message or "", is_path=False)
        return self._to_output(report)
