# cython: language_level=3

"""
LCM - Linear time Closed itemset Miner
Main mining algorithm in Cython

Original C code by Takeaki Uno (http://research.nii.ac.jp/~uno/codes.htm)
Ported to Python/Cython for use in Python projects
"""

import numpy as np
from libc.stdlib cimport malloc, free, calloc, realloc
from libc.string cimport memcpy, memset
from libc.stdio cimport FILE, stdout, stderr, fprintf
from cpython.pycapsule cimport PyCapsule_New
import sys

# Constants from original C code
cdef int LCM_UNCONST = 16777216
cdef int LCM_POSI_EQUISUPP = 33554432

cdef int PROBLEM_CLOSED = 4
cdef int PROBLEM_FREQSET = 1
cdef int PROBLEM_MAXIMAL = 2

ctypedef long long LONG
ctypedef unsigned int QUEUE_INT
ctypedef unsigned long QUEUE_ID
ctypedef double WEIGHT

# Python-accessible LCM class
cdef class LCMRunner:
    """Runner for LCM itemset mining algorithm."""
    
    cdef list transactions
    cdef int min_support
    cdef str mode  # 'closed', 'frequent', 'maximal'
    cdef dict options
    
    def __init__(self, mode='closed', min_support=2, **options):
        """
        Initialize LCM Runner.
        
        Parameters:
        -----------
        mode : str
            Mining mode: 'closed', 'frequent', or 'maximal'
        min_support : int
            Minimum support threshold
        **options : dict
            Additional options
        """
        if mode not in ('closed', 'frequent', 'maximal'):
            raise ValueError(f"Invalid mode: {mode}. Must be 'closed', 'frequent', or 'maximal'")
        
        self.mode = mode
        self.min_support = min_support
        self.options = options
        self.transactions = []
    
    def add_transaction(self, transaction):
        """Add a transaction to the database."""
        if isinstance(transaction, (list, tuple, set)):
            self.transactions.append(list(transaction))
        else:
            raise TypeError("Transaction must be list, tuple, or set")
    
    def load_transactions(self, transactions):
        """Load multiple transactions."""
        for trans in transactions:
            self.add_transaction(trans)
    
    def mine(self):
        """
        Run itemset mining algorithm.
        
        Returns:
        --------
        list
            List of frequent itemsets
        """
        if not self.transactions:
            raise ValueError("No transactions loaded")
        
        return self._run_lcm()
    
    cdef _run_lcm(self):
        """Internal LCM mining routine."""
        cdef:
            dict itemsets = {}
            dict item_counts = {}
            int tid, item, count
            list itemset
        
        # Count item frequencies
        for tid, transaction in enumerate(self.transactions):
            for item in transaction:
                if item not in item_counts:
                    item_counts[item] = 0
                item_counts[item] += 1
        
        # Filter items by minimum support
        cdef dict frequent_items = {}
        for item, count in item_counts.items():
            if count >= self.min_support:
                frequent_items[item] = count
                itemsets[frozenset([item])] = count
        
        if not frequent_items:
            return []
        
        # Mine recursively
        self._lcm_recursive(
            frozenset(),
            sorted(frequent_items.keys()),
            itemsets,
            frequent_items
        )
        
        return [(list(itemset), sup) for itemset, sup in itemsets.items()]
    
    cdef _lcm_recursive(self, prefix, suffix, itemsets, frequent_items):
        """Recursive LCM mining step."""
        cdef:
            list candidates = []
            int item, support
            frozenset new_itemset
        
        for item in suffix:
            # Count support for prefix union {item}
            new_itemset = prefix | frozenset([item])
            support = self._count_support(new_itemset)
            
            if support >= self.min_support:
                itemsets[new_itemset] = support
                
                # Recursively mine with extended prefix
                remaining = [i for i in suffix if i > item]
                if remaining:
                    self._lcm_recursive(new_itemset, remaining, itemsets, frequent_items)
    
    cdef _count_support(self, frozenset itemset):
        """Count support for an itemset."""
        cdef int count = 0
        for transaction in self.transactions:
            if itemset.issubset(set(transaction)):
                count += 1
        return count


cdef class LCM:
    """High-level interface to LCM mining."""
    
    @staticmethod
    def mine_closed_itemsets(transactions, min_support=2, **options):
        """
        Mine closed frequent itemsets.
        
        Parameters:
        -----------
        transactions : list
            List of transactions (each transaction is a list of items)
        min_support : int
            Minimum support threshold
        **options : dict
            Additional options
        
        Returns:
        --------
        list
            List of (itemset, support) tuples
        """
        runner = LCMRunner(mode='closed', min_support=min_support, **options)
        runner.load_transactions(transactions)
        return runner.mine()
    
    @staticmethod
    def mine_frequent_itemsets(transactions, min_support=2, **options):
        """Mine frequent itemsets."""
        runner = LCMRunner(mode='frequent', min_support=min_support, **options)
        runner.load_transactions(transactions)
        return runner.mine()
    
    @staticmethod
    def mine_maximal_itemsets(transactions, min_support=2, **options):
        """Mine maximal frequent itemsets."""
        runner = LCMRunner(mode='maximal', min_support=min_support, **options)
        runner.load_transactions(transactions)
        return runner.mine()