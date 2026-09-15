"""Unit tests for Milestone 2."""
import sys
from pathlib import Path

# Add milestone_2 root to sys.path so 'app' is importable from any working directory
milestone_2_root = str(Path(__file__).resolve().parent.parent)
if milestone_2_root not in sys.path:
    sys.path.insert(0, milestone_2_root)
