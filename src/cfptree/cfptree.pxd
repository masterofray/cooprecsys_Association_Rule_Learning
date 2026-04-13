# cython: language_level=3
"""CFP-tree declaration file."""

cdef class CFPtree:
    cdef unsigned int minsup_
    cdef list transactions
    cdef list header_table
    cdef dict conditional_db
    cdef int transaction_count
    cdef dict item_freq_map
    
    cdef void _remove_nonfreq_items_from_htable(self)
    cdef void _build_header_table(self)
    cdef void _construct_cond_db(self)