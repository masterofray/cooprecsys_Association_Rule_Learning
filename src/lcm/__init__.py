"""
LCM - Linear time Closed itemset Miner
Python/Cython port of the original C implementation by Takeaki Uno

This package provides frequent itemset mining, closed itemset mining,
and maximal itemset mining algorithms.
"""

from .lcm import LCM, LCMRunner
from .itemset import Itemset
from .problem import Problem

__version__ = "1.0.0"
__all__ = ["LCM", "LCMRunner", "Itemset", "Problem"]