"""Unit and integration tests for Milestone 4."""
import sys
from pathlib import Path

# Add milestone_4 root to sys.path so 'app' is importable from any working directory
milestone_4_root = str(Path(__file__).resolve().parent.parent)
if milestone_4_root not in sys.path:
    sys.path.insert(0, milestone_4_root)
