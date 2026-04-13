# cython: language_level=3

"""
Itemset operations in Cython
Ported from original C itemset.c/itemset.h
"""

from cpython cimport array
import array

ctypedef double WEIGHT
ctypedef unsigned int QUEUE_INT
ctypedef unsigned long QUEUE_ID
ctypedef long long LONG

cdef class Itemset:
    """Itemset structure and operations."""
    
    cdef list items
    cdef WEIGHT frequency
    cdef dict metadata
    
    def __init__(self, items=None, frequency=0.0):
        """Initialize itemset."""
        self.items = list(items) if items else []
        self.frequency = frequency
        self.metadata = {}
    
    def add_item(self, item):
        """Add item to itemset."""
        if item not in self.items:
            self.items.append(item)
    
    def remove_item(self, item):
        """Remove item from itemset."""
        if item in self.items:
            self.items.remove(item)
    
    def get_size(self):
        """Get itemset size."""
        return len(self.items)
    
    def get_items(self):
        """Get items in itemset."""
        return self.items.copy()
    
    def get_frequency(self):
        """Get frequency."""
        return self.frequency
    
    def set_frequency(self, WEIGHT freq):
        """Set frequency."""
        self.frequency = freq
    
    def set_metadata(self, key, value):
        """Set metadata."""
        self.metadata[key] = value
    
    def get_metadata(self, key):
        """Get metadata."""
        return self.metadata.get(key)
    
    def __repr__(self):
        return f"Itemset({self.items}, freq={self.frequency})"
    
    def __hash__(self):
        return hash(frozenset(self.items))
    
    def __eq__(self, other):
        if isinstance(other, Itemset):
            return set(self.items) == set(other.items)
        return False