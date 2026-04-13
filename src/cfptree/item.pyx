# cython: language_level=3
"""Item class for CFP-tree."""

cdef class Item:
    """Represents a single item in the CFP-tree."""
    
    cdef str item_
    cdef unsigned int freq_
    
    def __init__(self, str item, unsigned int freq=0):
        """Initialize an Item.
        
        Args:
            item: Item identifier string
            freq: Initial frequency count
        """
        self.item_ = item
        self.freq_ = freq
    
    def get_item(self) -> str:
        """Get the item identifier."""
        return self.item_
    
    def get_freq(self) -> int:
        """Get the frequency count."""
        return self.freq_
    
    def inc_freq(self, unsigned int n=1) -> None:
        """Increment frequency by n."""
        self.freq_ += n
    
    def set_freq(self, unsigned int n) -> None:
        """Set the frequency to n."""
        self.freq_ = n
    
    def __eq__(self, other):
        """Equality comparison."""
        if isinstance(other, str):
            return self.item_ == other
        elif isinstance(other, Item):
            return self.item_ == other.item_ and self.freq_ == other.freq_
        return False
    
    def __lt__(self, other):
        """Less than comparison by frequency."""
        if not isinstance(other, Item):
            return NotImplemented
        if self.freq_ == other.freq_:
            return self.item_ < other.item_
        return self.freq_ < other.freq_
    
    def __le__(self, other):
        """Less than or equal comparison."""
        return self < other or self == other
    
    def __gt__(self, other):
        """Greater than comparison."""
        if not isinstance(other, Item):
            return NotImplemented
        return other < self
    
    def __ge__(self, other):
        """Greater than or equal comparison."""
        return not (self < other)
    
    def __hash__(self):
        """Hash function for Item."""
        return hash(self.item_)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Item({self.item_}, freq={self.freq_})"
    
    def __str__(self) -> str:
        """Human-readable string."""
        return f"{self.item_}:{self.freq_}"