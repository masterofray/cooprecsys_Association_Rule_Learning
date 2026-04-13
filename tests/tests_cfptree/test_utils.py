"""Tests for utility functions."""

import pytest
import tempfile
import json
from pathlib import Path
from cfptree.utils import (
    load_transactions,
    save_results,
    load_results,
    format_results,
    generate_report,
)


class TestUtilsFunctions:
    """Test utility functions."""
    
    def test_load_transactions(self):
        """Test loading transactions from file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("a b c\n")
            f.write("a b\n")
            f.write("b c\n")
            temp_path = f.name
        
        try:
            transactions = load_transactions(temp_path)
            assert len(transactions) == 3
            assert transactions[0] == ['a', 'b', 'c']
            assert transactions[1] == ['a', 'b']
        finally:
            Path(temp_path).unlink()
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_transactions("/nonexistent/file.txt")
    
    def test_save_and_load_results(self):
        """Test saving and loading results."""
        results = {
            'min_support': 2,
            'items': {'a': 5, 'b': 3, 'c': 2},
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            save_results(results, temp_path)
            loaded = load_results(temp_path)
            assert loaded == results
        finally:
            Path(temp_path).unlink()
    
    def test_format_results(self):
        """Test formatting results."""
        patterns = {'a': 5, 'b': 3, 'c': 2, 'd': 1}
        formatted = format_results(patterns, min_freq=2)
        
        assert len(formatted) == 3
        assert formatted[0] == ('a', 5)
        assert formatted[1] == ('b', 3)
    
    def test_generate_report(self):
        """Test report generation."""
        stats = {
            'min_support': 2,
            'total_transactions': 10,
            'unique_items': 4,
            'frequent_items': 3,
            'avg_transaction_size': 2.5,
        }
        patterns = {'a': 5, 'b': 3, 'c': 2}
        
        report = generate_report(stats, patterns)
        
        assert 'CFP-TREE ANALYSIS REPORT' in report
        assert 'min_support: 2' in report
        assert 'a: 5' in report