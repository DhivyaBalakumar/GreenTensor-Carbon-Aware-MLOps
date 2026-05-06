"""
conftest.py — ensures the correct greentensor package is on sys.path
when running pytest from the workspace root or from this directory.
"""
import sys
import os

# Insert the package root (the directory containing the greentensor/ package)
# so that `import greentensor` resolves to greentensor/greentensor/__init__.py
_pkg_root = os.path.dirname(os.path.abspath(__file__))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)
