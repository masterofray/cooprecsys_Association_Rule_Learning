# cython: language_level=3

"""
Transaction handling in Cython
Ported from original C trsact.c/trsact.h
"""

from libc.stdlib cimport malloc, free
from cpython cimport array

ctypedef unsigned long QUEUE_ID
ctypedef double WEIGHT

cdef class Transaction:
    """Transaction structure."""
    
    cdef list items
    cdef WEIGHT weight
    cdef QUEUE_ID transaction_id
    
    def __init__(self, items, WEIGHT weight=1.0, QUEUE_ID tid=0):
        self.items = list(items)
        self.weight = weight
        self.transaction_id = tid
    
    def get_items(self):
        """Get transaction items."""
        return self.items.copy()
    
    def get_weight(self):
        """Get transaction weight."""
        return self.weight
    
    def get_id(self):
        """Get transaction ID."""
        return self.transaction_id
    
    def __repr__(self):
        return f"Transaction({self.items}, w={self.weight}, id={self.transaction_id})"

cdef class TransactionDatabase:
    """Transaction database structure."""
    
    cdef list transactions
    cdef WEIGHT total_weight
    
    def __init__(self):
        self.transactions = []
        self.total_weight = 0.0
    
    def add_transaction(self, transaction):
        """Add transaction to database."""
        if isinstance(transaction, Transaction):
            self.transactions.append(transaction)
            self.total_weight += transaction.get_weight()
        else:
            raise TypeError("Expected Transaction object")
    
    def get_transactions(self):
        """Get all transactions."""
        return self.transactions.copy()
    
    def get_transaction_count(self):
        """Get number of transactions."""
        return len(self.transactions)
    
    def get_total_weight(self):
        """Get total weight."""
        return self.total_weight