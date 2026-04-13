"""Visualization utilities for CFP-tree."""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from .cfptree import CFPtree


class Visualizer:
    """Visualization tools for CFP-tree analysis."""
    
    def __init__(self, cfptree: CFPtree):
        """Initialize visualizer.
        
        Args:
            cfptree: CFPtree instance to visualize
        """
        self.cfptree = cfptree
        sns.set_style("whitegrid")
    
    def plot_item_frequency_distribution(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """Plot 1: Item frequency distribution."""
        fig, ax = plt.subplots(figsize=figsize)
        
        patterns = self.cfptree.get_frequent_patterns()
        items = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
        
        if not items:
            ax.text(0.5, 0.5, 'No frequent items', ha='center', va='center')
            return fig
        
        items_names = [item[0] for item in items[:20]]
        items_freqs = [item[1] for item in items[:20]]
        
        bars = ax.bar(range(len(items_names)), items_freqs, color='steelblue')
        ax.set_xticks(range(len(items_names)))
        ax.set_xticklabels(items_names, rotation=45, ha='right')
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Items')
        ax.set_title('Item Frequency Distribution (Top 20)')
        ax.grid(axis='y', alpha=0.3)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def plot_cumulative_frequency(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """Plot 2: Cumulative frequency curve."""
        fig, ax = plt.subplots(figsize=figsize)
        
        patterns = self.cfptree.get_frequent_patterns()
        freqs = sorted(patterns.values(), reverse=True)
        cumsum = np.cumsum(freqs)
        
        if len(freqs) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            return fig
        
        cumsum_norm = cumsum / cumsum[-1] * 100
        ax.plot(range(len(cumsum_norm)), cumsum_norm, 'o-', linewidth=2, color='darkblue')
        ax.fill_between(range(len(cumsum_norm)), cumsum_norm, alpha=0.3)
        ax.set_xlabel('Items (sorted by frequency)')
        ax.set_ylabel('Cumulative Frequency (%)')
        ax.set_title('Cumulative Frequency Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_header_table_heatmap(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """Plot 3: Header table as heatmap."""
        fig, ax = plt.subplots(figsize=figsize)
        
        header_table = self.cfptree.get_header_table()
        
        if not header_table:
            ax.text(0.5, 0.5, 'No header table', ha='center', va='center')
            return fig
        
        items = [item[0] for item in header_table]
        freqs = np.array([item[1] for item in header_table]).reshape(-1, 1)
        
        im = ax.imshow(freqs, cmap='YlOrRd', aspect='auto')
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels(items)
        ax.set_xticks([0])
        ax.set_xticklabels(['Frequency'])
        ax.set_title('Header Table Frequencies')
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        return fig
    
    def plot_conditional_db_network(self, figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """Plot 4: Conditional database as network graph."""
        fig, ax = plt.subplots(figsize=figsize)
        
        G = nx.DiGraph()
        header_table = self.cfptree.get_header_table()
        
        for item, freq in header_table[:10]:  # Limit to top 10
            G.add_node(item, freq=freq)
            cond_db = self.cfptree.get_conditional_database(item)
            for suffix in cond_db[:5]:  # Limit suffixes
                for suffix_item in suffix:
                    if suffix_item != item:
                        G.add_edge(item, suffix_item)
        
        if len(G.nodes()) == 0:
            ax.text(0.5, 0.5, 'No conditional database', ha='center', va='center')
            return fig
        
        pos = nx.spring_layout(G, k=2, iterations=50)
        node_sizes = [G.nodes[node].get('freq', 10) * 10 for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color='lightblue', ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              arrows=True, ax=ax, arrowsize=15)
        
        ax.set_title('Conditional Database Network (Top 10 Items)')
        ax.axis('off')
        plt.tight_layout()
        return fig
    
    def plot_frequency_distribution_histogram(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """Plot 5: Frequency distribution histogram."""
        fig, ax = plt.subplots(figsize=figsize)
        
        patterns = self.cfptree.get_frequent_patterns()
        freqs = list(patterns.values())
        
        if not freqs:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            return fig
        
        ax.hist(freqs, bins=min(30, len(set(freqs))), color='coral', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Number of Items')
        ax.set_title('Distribution of Item Frequencies')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_statistics_summary(self, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """Plot 6: Statistics summary panel."""
        fig = plt.figure(figsize=figsize)
        
        stats = self.cfptree.get_statistics()
        
        # Create text summary
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        summary_text = (
            f"CFP-Tree Statistics\n"
            f"{'='*40}\n"
            f"Minimum Support: {stats['min_support']}\n"
            f"Total Transactions: {stats['total_transactions']}\n"
            f"Unique Items: {stats['unique_items']}\n"
            f"Frequent Items: {stats['frequent_items']}\n"
            f"Avg Transaction Size: {stats['avg_transaction_size']:.2f}\n"
            f"Frequent Item Ratio: {stats['frequent_items']/max(stats['unique_items'], 1)*100:.2f}%\n"
        )
        
        ax.text(0.5, 0.5, summary_text, fontsize=14, family='monospace',
               ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        return fig
    
    def plot_pareto_analysis(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """Plot 7: Pareto analysis (80/20 rule)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        patterns = self.cfptree.get_frequent_patterns()
        freqs = sorted(patterns.values(), reverse=True)
        
        if not freqs:
            ax1.text(0.5, 0.5, 'No data', ha='center', va='center')
            return fig
        
        total = sum(freqs)
        cumsum = np.cumsum(freqs) / total * 100
        
        # Pareto curve
        ax1.plot(range(len(cumsum)), cumsum, 'o-', linewidth=2, color='darkred')
        ax1.axhline(y=80, color='green', linestyle='--', label='80% threshold')
        ax1.axhline(y=20, color='blue', linestyle='--', label='20% threshold')
        ax1.set_xlabel('Items')
        ax1.set_ylabel('Cumulative Frequency %')
        ax1.set_title('Pareto Analysis')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Distribution
        ax2.bar(range(min(20, len(freqs))), freqs[:20], color='teal', alpha=0.7)
        ax2.set_xlabel('Item Rank')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Top 20 Items Distribution')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_item_pairs_matrix(self, figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """Plot 8: Item co-occurrence matrix."""
        fig, ax = plt.subplots(figsize=figsize)
        
        header_table = self.cfptree.get_header_table()[:15]  # Top 15 items
        item_list = [item[0] for item in header_table]
        
        if not item_list:
            ax.text(0.5, 0.5, 'No items', ha='center', va='center')
            return fig
        
        # Build co-occurrence matrix
        matrix = np.zeros((len(item_list), len(item_list)))
        for item, freq in header_table:
            cond_db = self.cfptree.get_conditional_database(item)
            for suffix in cond_db:
                for suffix_item in suffix:
                    if suffix_item in item_list:
                        idx_item = item_list.index(item)
                        idx_suffix = item_list.index(suffix_item)
                        matrix[idx_item, idx_suffix] += 1
        
        im = ax.imshow(matrix, cmap='Blues', aspect='auto')
        ax.set_xticks(range(len(item_list)))
        ax.set_yticks(range(len(item_list)))
        ax.set_xticklabels(item_list, rotation=45, ha='right')
        ax.set_yticklabels(item_list)
        ax.set_title('Item Co-occurrence Matrix (Top 15 Items)')
        
        plt.colorbar(im, ax=ax, label='Co-occurrence Count')
        plt.tight_layout()
        return fig
    
    def plot_frequency_vs_transactions(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """Plot 9: Frequency coverage analysis."""
        fig, ax = plt.subplots(figsize=figsize)
        
        patterns = self.cfptree.get_frequent_patterns()
        freqs = sorted(patterns.values(), reverse=True)
        
        if not freqs:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            return fig
        
        stats = self.cfptree.get_statistics()
        total_freq = sum(freqs)
        
        coverage = [(f / stats['total_transactions'] * 100) for f in freqs[:20]]
        
        ax.bar(range(len(coverage)), coverage, color='mediumseagreen', alpha=0.8)
        ax.set_xlabel('Item Rank')
        ax.set_ylabel('Coverage (%)')
        ax.set_title('Frequency Coverage by Transactions (Top 20)')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_support_levels(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """Plot 10: Support level distribution."""
        fig, ax = plt.subplots(figsize=figsize)
        
        header_table = self.cfptree.get_header_table()
        
        if not header_table:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            return fig
        
        items = [item[0] for item in header_table[:25]]
        freqs = [item[1] for item in header_table[:25]]
        minsup = self.cfptree.get_statistics()['min_support']
        
        colors = ['green' if f >= minsup else 'red' for f in freqs]
        bars = ax.barh(items, freqs, color=colors, alpha=0.7)
        
        ax.axvline(x=minsup, color='red', linestyle='--', linewidth=2, label=f'Minsup={minsup}')
        ax.set_xlabel('Support (Frequency)')
        ax.set_title('Support Levels for Frequent Items')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_all_visualizations(self, output_dir: str = './visualizations') -> None:
        """Generate and save all 10 visualizations.
        
        Args:
            output_dir: Directory to save visualizations
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        plots = [
            ('01_item_frequency_distribution', self.plot_item_frequency_distribution()),
            ('02_cumulative_frequency', self.plot_cumulative_frequency()),
            ('03_header_table_heatmap', self.plot_header_table_heatmap()),
            ('04_conditional_db_network', self.plot_conditional_db_network()),
            ('05_frequency_histogram', self.plot_frequency_distribution_histogram()),
            ('06_statistics_summary', self.plot_statistics_summary()),
            ('07_pareto_analysis', self.plot_pareto_analysis()),
            ('08_item_pairs_matrix', self.plot_item_pairs_matrix()),
            ('09_frequency_vs_transactions', self.plot_frequency_vs_transactions()),
            ('10_support_levels', self.plot_support_levels()),
        ]
        
        for name, fig in plots:
            filepath = os.path.join(output_dir, f'{name}.png')
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {filepath}")