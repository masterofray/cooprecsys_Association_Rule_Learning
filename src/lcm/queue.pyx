# cython: language_level=3

"""
Queue data structure in Cython
Ported from original C queue.c/queue.h
"""

from libc.stdlib cimport malloc, free, calloc, realloc
from libc.string cimport memcpy, memset

ctypedef unsigned int QUEUE_INT
ctypedef unsigned long QUEUE_ID

cdef class Queue:
    """Dynamic queue implementation."""
    
    cdef QUEUE_INT* v
    cdef QUEUE_ID s
    cdef QUEUE_ID t
    cdef QUEUE_ID end
    
    def __cinit__(self):
        self.v = NULL
        self.s = 0
        self.t = 0
        self.end = 0
    
    def __dealloc__(self):
        if self.v != NULL:
            free(self.v)
    
    cpdef alloc(self, QUEUE_ID size):
        """Allocate queue with given size."""
        if self.v != NULL:
            free(self.v)
        self.end = size
        self.v = <QUEUE_INT*>malloc(size * sizeof(QUEUE_INT))
        if self.v == NULL:
            raise MemoryError("Failed to allocate queue")
        self.s = 0
        self.t = 0
    
    cpdef push(self, QUEUE_INT item):
        """Push item to queue."""
        if self.t >= self.end:
            # Resize queue
            new_size = self.end * 2
            new_v = <QUEUE_INT*>realloc(self.v, new_size * sizeof(QUEUE_INT))
            if new_v == NULL:
                raise MemoryError("Failed to resize queue")
            self.v = new_v
            self.end = new_size
        
        self.v[self.t] = item
        self.t += 1
    
    cpdef QUEUE_INT pop(self):
        """Pop item from queue."""
        if self.t <= self.s:
            raise IndexError("Queue is empty")
        self.t -= 1
        return self.v[self.t]
    
    cpdef QUEUE_INT peek(self):
        """Peek at front item without removing."""
        if self.t <= self.s:
            raise IndexError("Queue is empty")
        return self.v[self.t - 1]
    
    cpdef bint is_empty(self):
        """Check if queue is empty."""
        return self.t <= self.s
    
    cpdef QUEUE_ID length(self):
        """Get queue length."""
        return self.t - self.s
    
    cpdef clear(self):
        """Clear queue."""
        self.s = 0
        self.t = 0
    
    cpdef list to_python_list(self):
        """Convert queue to Python list."""
        cdef list result = []
        cdef QUEUE_ID i
        for i in range(self.s, self.t):
            result.append(self.v[i])
        return result