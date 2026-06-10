"""Pytest configuration for the codeanalysis datasource test suite.

Adds the plugin root to ``sys.path`` so ``source_code_ai_detector`` and
``SubjectiveCodeanalysisDataSource`` import without installation.
"""

import os
import sys

PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, PLUGIN_ROOT)

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
