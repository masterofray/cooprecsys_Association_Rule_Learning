# distutils: language = c
# cython: language_level=3, boundscheck=False, wraparound=False

from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memcpy, strcpy
from libc.stdio cimport fprintf, stdout, FILE, fopen, fclose
import numpy as np
cimport numpy as np
from cpython cimport PyObject
from cython.parallel cimport prange
import threading
from collections import defaultdict

ctypedef np.int32_t DTYPE_int
ctypedef np.float64_t DTYPE_float

cdef extern from "stdio.h":
    int printf(const char* format, ...)

cdef class FPNode:
    """Cython implementation of FP-Tree Node"""
    cdef public int item
    cdef public long count
    cdef public FPNode parent
    cdef public dict children
    cdef public list link
    
    def __init__(self, int item=-1, long count=0, FPNode parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.link = []
    
    cdef void add_child(self, FPNode child):
        """Add a child node"""
        self.children[child.item] = child
    
    cdef FPNode get_child(self, int item):
        """Get child by item"""
        return self.children.get(item, None)
    
    cdef list get_path_to_root(self):
        """Get path from this node to root"""
        cdef list path = []
        cdef FPNode node = self.parent
        
        while node is not None and node.item != -1:
            path.append(node.item)
            node = node.parent
        
        path.reverse()
        return path


cdef class FPTree:
    """Cython implementation of FP-Tree"""
    cdef public FPNode root
    cdef public dict header_table
    cdef public dict rank
    cdef public list cond_items
    
    def __init__(self, dict rank=None):
        self.root = FPNode(-1, 0, None)
        self.header_table = defaultdict(list)
        self.rank = rank if rank is not None else {}
        self.cond_items = []
    
    cdef void insert_itemset(self, list itemset, long count=1):
        """Insert itemset into the tree"""
        cdef FPNode node = self.root
        cdef FPNode child
        cdef int item
        
        node.count += count
        
        if len(itemset) == 0:
            return
        
        # Follow existing path
        cdef int index = 0
        for item in itemset:
            child = node.get_child(item)
            if child is not None:
                child.count += count
                node = child
                index += 1
            else:
                break
        
        # Insert remaining items
        for item in itemset[index:]:
            child_node = FPNode(item, count, node)
            node.add_child(child_node)
            self.header_table[item].append(child_node)
            node = child_node
    
    cdef bint is_path(self):
        """Check if tree is a path"""
        if len(self.root.children) > 1:
            return False
        
        for item in self.header_table:
            if len(self.header_table[item]) > 1:
                return False
            if len(self.header_table[item][0].children) > 1:
                return False
        
        return True
    
    cdef FPTree conditional_tree(self, int cond_item, long minsup):
        """Create conditional tree"""
        cdef FPTree cond_tree = FPTree()
        cdef dict count = defaultdict(int)
        cdef list branches = []
        cdef FPNode node
        cdef list branch
        cdef int item
        
        fprintf(stdout, "Building conditional tree for item: %d\n", cond_item)
        
        # Find all paths
        for node in self.header_table.get(cond_item, []):
            branch = node.get_path_to_root()
            branches.append((branch, node.count))
            
            for item in branch:
                count[item] += node.count
        
        # Filter items by minsup
        cdef list items = [item for item in count if count[item] >= minsup]
        items.sort(key=count.get)
        cdef dict new_rank = {item: i for i, item in enumerate(items)}
        
        # Build conditional tree
        cdef list sorted_branch
        for branch, node_count in branches:
            sorted_branch = sorted(
                [i for i in branch if i in new_rank],
                key=new_rank.get,
                reverse=True
            )
            cond_tree.insert_itemset(sorted_branch, node_count)
        
        cond_tree.rank = new_rank
        cond_tree.cond_items = self.cond_items + [cond_item]
        
        return cond_tree


cdef class FPGrowthCython:
    """Production-grade FPGrowth implementation in Cython"""
    cdef public dict frequent_itemsets
    cdef public long min_support_count
    cdef public long total_transactions
    cdef public FPTree tree
    cdef public dict rank
    
    def __init__(self):
        self.frequent_itemsets = {}
        self.min_support_count = 0
        self.total_transactions = 0
        self.tree = None
        self.rank = {}
    
    def fit(self, np.ndarray[DTYPE_int, ndim=2] data, double min_support):
        """
        Fit FPGrowth model to data
        
        Parameters
        ----------
        data : np.ndarray
            Binary matrix (n_transactions, n_items)
        min_support : float
            Minimum support threshold
        """
        cdef long n_transactions = data.shape[0]
        cdef int n_items = data.shape[1]
        cdef long min_sup_count = <long>np.ceil(min_support * n_transactions)
        cdef np.ndarray[DTYPE_int, ndim=1] item_support = np.zeros(n_items, dtype=np.int32)
        cdef int i, j, item
        cdef list itemset
        
        fprintf(stdout, "FPGrowth: Starting fit with %ld transactions\n", n_transactions)
        fprintf(stdout, "FPGrowth: Min support count = %ld\n", min_sup_count)
        
        self.total_transactions = n_transactions
        self.min_support_count = min_sup_count
        
        # Calculate item support
        for j in range(n_items):
            item_support[j] = 0
            for i in range(n_transactions):
                item_support[j] += data[i, j]
        
        # Filter frequent items
        cdef list frequent_items = []
        for j in range(n_items):
            if item_support[j] >= min_sup_count:
                frequent_items.append((j, item_support[j]))
        
        fprintf(stdout, "FPGrowth: Found %ld frequent items\n", len(frequent_items))
        
        # Sort by support
        frequent_items.sort(key=lambda x: x[1])
        self.rank = {item: i for i, (item, _) in enumerate(frequent_items)}
        
        # Build FP-tree
        self.tree = FPTree(self.rank)
        for i in range(n_transactions):
            itemset = [j for j in range(n_items) if data[i, j] == 1 and j in self.rank]
            itemset.sort(key=lambda x: self.rank[x], reverse=True)
            self.tree.insert_itemset(itemset)
        
        fprintf(stdout, "FPGrowth: FP-tree built successfully\n")
    
    def get_frequent_itemsets(self, dict max_len=None):
        """Get all frequent itemsets"""
        if self.tree is None:
            raise ValueError("Model must be fitted first")
        
        cdef list itemsets = []
        cdef list supports = []
        self._fpg_step(self.tree, itemsets, supports, max_len)
        
        return itemsets, supports
    
    cdef void _fpg_step(self, FPTree tree, list itemsets, list supports, dict max_len):
        """Recursive FPGrowth step"""
        cdef int count = 0
        cdef list items = list(tree.header_table.keys())
        cdef int item
        cdef long support
        cdef FPTree cond_tree
        
        fprintf(stdout, "FPGrowth: Processing %d items in tree\n", len(items))
        
        if tree.is_path():
            # Combinatorial generation for paths
            from itertools import combinations
            cdef int size_remain = len(items) + 1
            
            for i in range(1, size_remain):
                for itemset_combo in combinations(items, i):
                    support = min([node.count for node in tree.header_table[item] for item in itemset_combo])
                    itemsets.append(tree.cond_items + list(itemset_combo))
                    supports.append(support / <double>self.total_transactions)
        else:
            # Standard FPGrowth
            for item in items:
                support = sum([node.count for node in tree.header_table[item]])
                itemsets.append(tree.cond_items + [item])
                supports.append(support / <double>self.total_transactions)
                count += 1
        
        # Generate conditional trees
        if not tree.is_path():
            for item in items:
                cond_tree = tree.conditional_tree(item, self.min_support_count)
                if len(cond_tree.header_table) > 0:
                    self._fpg_step(cond_tree, itemsets, supports, max_len)