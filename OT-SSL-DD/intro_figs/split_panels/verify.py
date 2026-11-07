"""Compatibility entry point; settings and checks live in generate_figures.py."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generate_figures import verify_outputs

if __name__ == '__main__':
    verify_outputs()
