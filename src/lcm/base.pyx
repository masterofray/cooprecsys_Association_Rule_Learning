# cython: language_level=3

"""
Base memory management in Cython
Ported from original C base.c/base.h
"""

from libc.stdlib cimport malloc, free, calloc, realloc
from libc.string cimport memcpy, memset

cdef int BASE_UNIT = 16
cdef int BASE_BLOCK = 65536

cdef class MemoryPool:
    """Memory pool for efficient block allocation."""
    
    cdef char** blocks
    cdef int block_count
    cdef int block_size
    cdef int unit_size
    cdef int current_block
    cdef int current_position
    
    def __cinit__(self, int unit_size, int block_size):
        self.unit_size = unit_size
        self.block_size = block_size
        self.block_count = 0
        self.current_block = -1
        self.current_position = 0
        self.blocks = NULL
    
    def __dealloc__(self):
        cdef int i
        if self.blocks != NULL:
            for i in range(self.block_count):
                if self.blocks[i] != NULL:
                    free(self.blocks[i])
            free(self.blocks)
    
    cdef void* allocate(self):
        """Allocate memory from pool."""
        if self.current_block == -1 or self.current_position >= self.block_size:
            self._add_block()
        
        cdef char* ptr = self.blocks[self.current_block] + self.current_position
        self.current_position += self.unit_size
        return ptr
    
    cdef void _add_block(self):
        """Add a new block to the pool."""
        cdef char** new_blocks
        cdef char* new_block
        
        new_blocks = <char**>realloc(self.blocks, (self.block_count + 1) * sizeof(char*))
        if new_blocks == NULL:
            raise MemoryError("Failed to allocate memory")
        
        self.blocks = new_blocks
        new_block = <char*>malloc(self.block_size)
        if new_block == NULL:
            raise MemoryError("Failed to allocate memory block")
        
        self.blocks[self.block_count] = new_block
        self.block_count += 1
        self.current_block = self.block_count - 1
        self.current_position = 0