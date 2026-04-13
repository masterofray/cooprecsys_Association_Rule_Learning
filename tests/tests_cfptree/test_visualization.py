"""Tests for visualization module."""

import pytest
import matplotlib.pyplot as plt
from cfptree import CFPtree, Visualizer


@pytest.fixture
def cfptree_with_data():
    """Create a CFP-tree with sample data."""
    cfptree = CFPtree(minsup=2)
    transactions = [
        ['apple', 'banana', 'cherry'],
        ['apple', 'banana'],
        ['apple', 'cherry'],
        ['banana', 'cherry'],
        ['apple', 'banana', 'cherry', 'date'],
        ['apple', 'banana'],
        ['banana', 'cherry'],
        ['apple', 'cherry'],
        ['cherry', 'date'],
        ['apple', 'date'],
    ]
    cfptree.read_transactions_from_list(transactions)
    cfptree.construct()
    return cfptree


class TestVisualizer:
    """Test visualization functions."""
    
    def test_visualizer_initialization(self, cfptree_with_data):
        """Test visualizer initialization."""
        visualizer = Visualizer(cfptree_with_data)
        assert visualizer.cfptree is cfptree_with_data
    
    def test_plot_item_frequency_distribution(self, cfptree_with_data):
        """Test item frequency distribution plot."""
        visualizer = Visualizer(cfptree_with_data)
        fig = visualizer.plot_item_frequency_distribution()
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_cumulative_frequency(self, cfptree_with_data):
        """Test cumulative frequency plot."""
        visualizer = Visualizer(cfptree_with_data)
        fig = visualizer.plot_cumulative_frequency()
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_header_table_heatmap(self, cfptree_with_data):
        """Test header table heatmap plot."""
        visualizer = Visualizer(cfptree_with_data)
        fig = visualizer.plot_header_table_heatmap()
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_conditional_db_network(self, cfptree_with_data):
        """Test conditional database network plot."""
        visualizer = Visualizer(cfptree_with_data)
        fig = visualizer.plot_conditional_db_network()
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_frequency_distribution_histogram(self, cfptree_with_data):
        """Test frequency distribution histogram."""
        visualizer = Visualizer(cfptree_with_data)
        fig = visualizer.plot_frequency_distribution_histogram()
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_statistics_summary(self, cfptree_with_data):
        """Test statistics summary plot."""
        visualizer = Visualizer(cfptree_with_data)
        fig = visualizer.plot_statistics_summary()
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_pareto_analysis(self, cfptree_with_data):
        """Test pareto analysis plot."""
        visualizer = Visualizer(cfptree_with_data)
        fig = visualizer.plot_pareto_analysis()
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_item_pairs_matrix(self, cfptree_with_data):
        """Test item pairs matrix plot."""
        visualizer = Visualizer(cfptree_with_data)
        fig = visualizer.plot_item_pairs_matrix()
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_frequency_vs_transactions(self, cfptree_with_data):
        """Test frequency vs transactions plot."""
        visualizer = Visualizer(cfptree_with_data)
        fig = visualizer.plot_frequency_vs_transactions()
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_support_levels(self, cfptree_with_data):
        """Test support levels plot."""
        visualizer = Visualizer(cfptree_with_data)
        fig = visualizer.plot_support_levels()
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_generate_all_visualizations(self, cfptree_with_data, tmp_path):
        """Test generating all visualizations."""
        visualizer = Visualizer(cfptree_with_data)
        output_dir = tmp_path / "visualizations"
        
        visualizer.generate_all_visualizations(str(output_dir))
        
        # Check that files were created
        files = list(output_dir.glob("*.png"))
        assert len(files) == 10