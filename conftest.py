"""
conftest.py — ensures the correct greentensor package is on sys.path
when running pytest from the workspace root or from this directory.
"""
import sys
import os

# Insert the package root so that `import greentensor` resolves correctly
# whether running locally or in CI with pip install -e .
_pkg_root = os.path.dirname(os.path.abspath(__file__))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)
