"""Unit tests for Milestone 1."""
import sys
from pathlib import Path

# Add milestone_1 root to sys.path so 'app' is importable from any working directory
milestone_1_root = str(Path(__file__).resolve().parent.parent)
if milestone_1_root not in sys.path:
    sys.path.insert(0, milestone_1_root)
