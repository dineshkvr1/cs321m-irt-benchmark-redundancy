"""Adaptive labeling: random baseline (no special acquisition strategy)."""
from __future__ import annotations
import random

def acquisition_function(input: dict) -> float:
    return random.random()
