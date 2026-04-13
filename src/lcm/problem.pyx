# cython: language_level=3

"""
Problem management in Cython
Ported from original C problem.c/problem.h
"""

from .itemset cimport Itemset
from .trsact cimport TransactionDatabase

ctypedef int QUEUE_INT

cdef int PROBLEM_FREQSET = 1
cdef int PROBLEM_MAXIMAL = 2
cdef int PROBLEM_CLOSED = 4

cdef class Problem:
    """Problem definition and management."""
    
    cdef int problem_type
    cdef int min_support
    cdef int max_itemset_size
    cdef int min_itemset_size
    cdef TransactionDatabase database
    cdef dict options
    
    def __init__(self, problem_type='closed', min_support=2):
        """Initialize problem."""
        if problem_type == 'closed':
            self.problem_type = PROBLEM_CLOSED
        elif problem_type == 'frequent':
            self.problem_type = PROBLEM_FREQSET
        elif problem_type == 'maximal':
            self.problem_type = PROBLEM_MAXIMAL
        else:
            raise ValueError(f"Invalid problem type: {problem_type}")
        
        self.min_support = min_support
        self.max_itemset_size = -1
        self.min_itemset_size = 1
        self.database = TransactionDatabase()
        self.options = {}
    
    def set_support(self, int support):
        """Set minimum support."""
        self.min_support = support
    
    def set_max_size(self, int size):
        """Set maximum itemset size."""
        self.max_itemset_size = size
    
    def set_min_size(self, int size):
        """Set minimum itemset size."""
        self.min_itemset_size = size
    
    def get_problem_type(self):
        """Get problem type."""
        return self.problem_type
    
    def get_database(self):
        """Get transaction database."""
        return self.database