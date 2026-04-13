# cython: language_level=3
"""CFP-tree implementation in Cython."""

from libc.stdlib cimport malloc, free
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.object cimport PyObject
from cfptree.item cimport Item
import numpy as np
from typing import List, Dict, Set, Tuple, Optional

cdef class CFPtree:
    """Conditional Frequent Pattern Tree for efficient pattern mining."""
    
    cdef unsigned int minsup_
    cdef list transactions
    cdef list header_table
    cdef dict conditional_db
    cdef int transaction_count
    cdef dict item_freq_map
    
    def __init__(self, unsigned int minsup):
        """Initialize CFP-tree.
        
        Args:
            minsup: Minimum support threshold
        """
        self.minsup_ = minsup
        self.transactions = []
        self.header_table = []
        self.conditional_db = {}
        self.transaction_count = 0
        self.item_freq_map = {}
    
    def read_transactions(self, str filename: str) -> int:
        """Read transactions from file.
        
        Args:
            filename: Path to transaction file
            
        Returns:
            Number of transactions read
        """
        cdef list tmp
        cdef str line, item
        cdef Item item_obj
        cdef int count = 0
        
        try:
            with open(filename, 'r') as f:
                for line in f:
                    tmp = []
                    items_in_line = set()
                    
                    for item in line.strip().split():
                        if item not in items_in_line:
                            items_in_line.add(item)
                            tmp.append(Item(item, 1))
                            
                            # Update header table
                            if item in self.item_freq_map:
                                self.item_freq_map[item] += 1
                            else:
                                self.item_freq_map[item] = 1
                    
                    if tmp:
                        self.transactions.append(tmp)
                        count += 1
            
            self.transaction_count = count
            return count
        
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filename}")
    
    def read_transactions_from_list(self, transactions: List[List[str]]) -> int:
        """Read transactions from list of lists.
        
        Args:
            transactions: List of transaction item lists
            
        Returns:
            Number of transactions read
        """
        cdef list tmp
        cdef str item
        cdef int count = 0
        
        for transaction in transactions:
            tmp = []
            items_in_line = set()
            
            for item in transaction:
                if item not in items_in_line:
                    items_in_line.add(item)
                    tmp.append(Item(item, 1))
                    
                    if item in self.item_freq_map:
                        self.item_freq_map[item] += 1
                    else:
                        self.item_freq_map[item] = 1
            
            if tmp:
                self.transactions.append(tmp)
                count += 1
        
        self.transaction_count = count
        return count
    
    def construct(self) -> None:
        """Construct the CFP-tree."""
        self._remove_nonfreq_items_from_htable()
        self._construct_cond_db()
        self._build_header_table()
    
    cdef void _remove_nonfreq_items_from_htable(self):
        """Remove items with support below minsup."""
        items_to_remove = []
        for item, freq in self.item_freq_map.items():
            if freq < self.minsup_:
                items_to_remove.append(item)
        
        for item in items_to_remove:
            del self.item_freq_map[item]
    
    cdef void _build_header_table(self):
        """Build sorted header table."""
        self.header_table = sorted(
            [Item(item, freq) for item, freq in self.item_freq_map.items()],
            key=lambda x: (x.get_freq(), x.get_item())
        )
    
    cdef void _construct_cond_db(self):
        """Construct conditional database."""
        for transaction in self.transactions:
            # Filter non-frequent items
            filtered = []
            for item_obj in transaction:
                if item_obj.get_item() in self.item_freq_map:
                    filtered.append(item_obj)
            
            if not filtered:
                continue
            
            # Sort by frequency
            filtered.sort()
            
            # Add to conditional database
            if filtered:
                prefix_item = filtered[0].get_item()
                suffix = [item_obj.get_item() for item_obj in filtered[1:]]
                
                if prefix_item not in self.conditional_db:
                    self.conditional_db[prefix_item] = []
                
                self.conditional_db[prefix_item].append(suffix)
    
    def get_frequent_patterns(self, unsigned int min_support=None) -> Dict[str, int]:
        """Get all frequent patterns.
        
        Args:
            min_support: Optional minimum support override
            
        Returns:
            Dictionary of patterns and their frequencies
        """
        if min_support is None:
            min_support = self.minsup_
        
        return dict(self.item_freq_map.copy())
    
    def get_header_table(self) -> List[Tuple[str, int]]:
        """Get the sorted header table.
        
        Returns:
            List of (item, frequency) tuples
        """
        return [(item.get_item(), item.get_freq()) for item in self.header_table]
    
    def get_conditional_database(self, str item) -> List[List[str]]:
        """Get conditional database for an item.
        
        Args:
            item: Item to get conditional DB for
            
        Returns:
            List of transaction suffixes
        """
        return self.conditional_db.get(item, [])
    
    def get_statistics(self) -> Dict:
        """Get tree statistics.
        
        Returns:
            Dictionary with tree statistics
        """
        return {
            'min_support': self.minsup_,
            'total_transactions': self.transaction_count,
            'unique_items': len(self.item_freq_map),
            'frequent_items': len(self.header_table),
            'avg_transaction_size': sum(len(t) for t in self.transactions) / max(self.transaction_count, 1),
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return f"CFPtree(minsup={self.minsup_}, transactions={self.transaction_count}, items={len(self.item_freq_map)})"