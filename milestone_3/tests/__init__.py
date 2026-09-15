"""Unit tests for Milestone 3."""
import sys
from pathlib import Path

# Add milestone_3 root to sys.path so 'app' is importable from any working directory
milestone_3_root = str(Path(__file__).resolve().parent.parent)
if milestone_3_root not in sys.path:
    sys.path.insert(0, milestone_3_root)
