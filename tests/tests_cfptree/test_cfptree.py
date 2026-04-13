"""Tests for CFP-tree core functionality."""

import pytest
import tempfile
from pathlib import Path
from cfptree import CFPtree


class TestCFPtreeBasic:
    """Test basic CFP-tree operations."""
    
    @pytest.fixture
    def cfptree(self):
        """Create a CFP-tree instance."""
        return CFPtree(minsup=2)
    
    @pytest.fixture
    def sample_transactions(self):
        """Sample transactions."""
        return [
            ['a', 'b', 'c'],
            ['a', 'b'],
            ['a', 'c'],
            ['b', 'c'],
            ['a', 'b', 'c', 'd'],
        ]
    
    def test_initialization(self, cfptree):
        """Test CFP-tree initialization."""
        assert cfptree.minsup_ == 2
        assert cfptree.transaction_count == 0
    
    def test_read_transactions_from_list(self, cfptree, sample_transactions):
        """Test reading transactions from list."""
        count = cfptree.read_transactions_from_list(sample_transactions)
        assert count == len(sample_transactions)
        assert cfptree.transaction_count == len(sample_transactions)
    
    def test_construct(self, cfptree, sample_transactions):
        """Test tree construction."""
        cfptree.read_transactions_from_list(sample_transactions)
        cfptree.construct()
        
        patterns = cfptree.get_frequent_patterns()
        assert len(patterns) > 0
    
    def test_get_statistics(self, cfptree, sample_transactions):
        """Test statistics retrieval."""
        cfptree.read_transactions_from_list(sample_transactions)
        cfptree.construct()
        
        stats = cfptree.get_statistics()
        assert 'min_support' in stats
        assert 'total_transactions' in stats
        assert 'unique_items' in stats
        assert 'frequent_items' in stats
        assert stats['min_support'] == 2
        assert stats['total_transactions'] == len(sample_transactions)
    
    def test_get_header_table(self, cfptree, sample_transactions):
        """Test header table retrieval."""
        cfptree.read_transactions_from_list(sample_transactions)
        cfptree.construct()
        
        header_table = cfptree.get_header_table()
        assert len(header_table) > 0
        assert all(isinstance(item, tuple) and len(item) == 2 for item in header_table)
    
    def test_get_conditional_database(self, cfptree, sample_transactions):
        """Test conditional database retrieval."""
        cfptree.read_transactions_from_list(sample_transactions)
        cfptree.construct()
        
        # Get an item from header table
        header_table = cfptree.get_header_table()
        if header_table:
            item = header_table[0][0]
            cond_db = cfptree.get_conditional_database(item)
            assert isinstance(cond_db, list)
    
    def test_str_representation(self, cfptree, sample_transactions):
        """Test string representation."""
        cfptree.read_transactions_from_list(sample_transactions)
        cfptree.construct()
        
        str_repr = str(cfptree)
        assert 'CFPtree' in str_repr
        assert 'minsup' in str_repr


class TestCFPtreeFileIO:
    """Test file I/O operations."""
    
    def test_read_transactions_from_file(self):
        """Test reading transactions from file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("a b c\n")
            f.write("a b\n")
            f.write("b c\n")
            f.write("a c\n")
            temp_path = f.name
        
        try:
            cfptree = CFPtree(minsup=2)
            count = cfptree.read_transactions(temp_path)
            assert count == 4
            
            cfptree.construct()
            patterns = cfptree.get_frequent_patterns()
            assert len(patterns) > 0
        finally:
            Path(temp_path).unlink()
    
    def test_read_nonexistent_file(self):
        """Test reading non-existent file."""
        cfptree = CFPtree(minsup=2)
        with pytest.raises(FileNotFoundError):
            cfptree.read_transactions("/nonexistent/path/file.txt")


class TestCFPtreeEdgeCases:
    """Test edge cases."""
    
    def test_empty_transactions(self):
        """Test with empty transactions list."""
        cfptree = CFPtree(minsup=2)
        count = cfptree.read_transactions_from_list([])
        assert count == 0
    
    def test_single_transaction(self):
        """Test with single transaction."""
        cfptree = CFPtree(minsup=1)
        cfptree.read_transactions_from_list([['a', 'b', 'c']])
        cfptree.construct()
        
        stats = cfptree.get_statistics()
        assert stats['total_transactions'] == 1
    
    def test_duplicate_items_in_transaction(self):
        """Test transaction with duplicate items."""
        cfptree = CFPtree(minsup=1)
        cfptree.read_transactions_from_list([['a', 'a', 'b', 'b']])
        cfptree.construct()
        
        patterns = cfptree.get_frequent_patterns()
        # Duplicates should be counted only once per transaction
        assert patterns.get('a') == 1
    
    def test_high_minsup(self):
        """Test with high minimum support."""
        cfptree = CFPtree(minsup=100)
        cfptree.read_transactions_from_list([
            ['a', 'b'], ['a', 'b'], ['c', 'd']
        ])
        cfptree.construct()
        
        patterns = cfptree.get_frequent_patterns()
        # No items should meet the high threshold
        assert len(patterns) == 0 or all(v >= 100 for v in patterns.values())


class TestCFPtreeStatistical:
    """Test statistical properties."""
    
    def test_avg_transaction_size(self):
        """Test average transaction size calculation."""
        cfptree = CFPtree(minsup=1)
        transactions = [
            ['a', 'b', 'c'],  # size 3
            ['a', 'b'],       # size 2
            ['a'],            # size 1
        ]
        cfptree.read_transactions_from_list(transactions)
        cfptree.construct()
        
        stats = cfptree.get_statistics()
        expected_avg = (3 + 2 + 1) / 3
        assert abs(stats['avg_transaction_size'] - expected_avg) < 0.01
    
    def test_frequent_item_count(self):
        """Test frequent item counting."""
        cfptree = CFPtree(minsup=2)
        transactions = [
            ['a', 'b'],
            ['a', 'b'],
            ['a', 'c'],
        ]
        cfptree.read_transactions_from_list(transactions)
        cfptree.construct()
        
        header_table = cfptree.get_header_table()
        # 'a' and 'b' have freq >= 2, 'c' has freq 1
        assert len(header_table) >= 2
        assert all(freq >= 2 for _, freq in header_table)