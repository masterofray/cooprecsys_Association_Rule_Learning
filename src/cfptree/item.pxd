# cython: language_level=3
"""Item declaration file."""

cdef class Item:
    cdef str item_
    cdef unsigned int freq_
    
    cpdef str get_item(self)
    cpdef unsigned int get_freq(self)
    cpdef void inc_freq(self, unsigned int n=*)
    cpdef void set_freq(self, unsigned int n)