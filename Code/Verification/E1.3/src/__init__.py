# E1.3 Experiment Package
"""
Experiment E1.3: PriceMoE MVP & Variable Top-k (E1.2 Compatible)

This package provides tools for running and analyzing PriceMoE experiments.
"""

from pathlib import Path

# Package root
PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent.parent.parent  # Code/Verification/E1.3 -> Project-MoE

__all__ = ["PACKAGE_ROOT", "PROJECT_ROOT"]
