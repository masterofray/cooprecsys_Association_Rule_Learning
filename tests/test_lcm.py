"""
Unit tests for LCM mining functionality.
"""

import pytest
from lcm import LCM, LCMRunner

class TestLCMMining:
    """Test LCM mining algorithms."""
    
    @pytest.fixture
    def sample_transactions(self):
        """Sample transaction database."""
        return [
            [1, 2, 3],
            [1, 2],
            [1, 3],
            [2, 3],
            [1, 2, 3, 4],
        ]
    
    def test_closed_itemset_mining(self, sample_transactions):
        """Test closed itemset mining."""
        itemsets = LCM.mine_closed_itemsets(sample_transactions, min_support=2)
        assert len(itemsets) > 0
        assert all(isinstance(items, list) for items, _ in itemsets)
    
    def test_frequent_itemset_mining(self, sample_transactions):
        """Test frequent itemset mining."""
        itemsets = LCM.mine_frequent_itemsets(sample_transactions, min_support=2)
        assert len(itemsets) > 0
    
    def test_maximal_itemset_mining(self, sample_transactions):
        """Test maximal itemset mining."""
        itemsets = LCM.mine_maximal_itemsets(sample_transactions, min_support=2)
        assert len(itemsets) > 0
    
    def test_support_threshold(self, sample_transactions):
        """Test support threshold filtering."""
        itemsets_2 = LCM.mine_closed_itemsets(sample_transactions, min_support=2)
        itemsets_3 = LCM.mine_closed_itemsets(sample_transactions, min_support=3)
        assert len(itemsets_3) <= len(itemsets_2)
    
    def test_lcm_runner(self, sample_transactions):
        """Test LCMRunner."""
        runner = LCMRunner(mode='closed', min_support=2)
        runner.load_transactions(sample_transactions)
        results = runner.mine()
        assert len(results) > 0
    
    def test_empty_database(self):
        """Test with empty transaction database."""
        with pytest.raises(ValueError):
            runner = LCMRunner()
            runner.mine()
    
    def test_single_item_transactions(self):
        """Test with single item transactions."""
        transactions = [[1], [2], [3], [1], [2]]
        itemsets = LCM.mine_closed_itemsets(transactions, min_support=2)
        # Should find [1] and [2] with support 2
        assert len(itemsets) >= 2

class TestLCMRunner:
    """Test LCMRunner class."""
    
    def test_add_transaction(self):
        """Test adding transactions."""
        runner = LCMRunner()
        runner.add_transaction([1, 2, 3])
        assert len(runner.transactions) == 1
    
    def test_load_transactions(self):
        """Test loading multiple transactions."""
        runner = LCMRunner()
        transactions = [[1, 2], [2, 3], [1, 3]]
        runner.load_transactions(transactions)
        assert len(runner.transactions) == 3
    
    def test_invalid_mode(self):
        """Test invalid mining mode."""
        with pytest.raises(ValueError):
            LCMRunner(mode='invalid')
    
    def test_mode_persistence(self):
        """Test that mining mode persists."""
        runner = LCMRunner(mode='maximal', min_support=3)
        assert runner.mode == 'maximal'
        assert runner.min_support == 3

if __name__ == "__main__":
    pytest.main([__file__, "-v"])