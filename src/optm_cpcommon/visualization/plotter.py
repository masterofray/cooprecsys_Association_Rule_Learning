"""
Production-grade visualization module with 15+ plots
"""

import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

logger = logging.getLogger(__name__)


class RecommenderPlotter:
    """Generate comprehensive visualizations for FPGrowth recommender"""
    
    def __init__(self, output_dir: str = "./plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")
        self.dpi = 100
    
    def plot_all(self, frequent_itemsets: pd.DataFrame, 
                 association_rules: pd.DataFrame, binary_matrix: np.ndarray):
        """Generate all visualizations"""
        logger.info("Starting visualization generation...")
        
        # 1. Support Distribution
        self._plot_support_distribution(frequent_itemsets)
        
        # 2. Support vs Itemset Size
        self._plot_support_vs_size(frequent_itemsets)
        
        # 3. Top 20 Itemsets
        self._plot_top_itemsets(frequent_itemsets, top_k=20)
        
        # 4. Itemset Size Distribution
        self._plot_itemset_size_distribution(frequent_itemsets)
        
        # 5. Confidence Distribution
        self._plot_confidence_distribution(association_rules)
        
        # 6. Lift Distribution
        self._plot_lift_distribution(association_rules)
        
        # 7. Confidence vs Lift
        self._plot_confidence_vs_lift(association_rules)
        
        # 8. Support vs Confidence
        self._plot_support_vs_confidence(association_rules)
        
        # 9. Rules Heatmap
        self._plot_rules_heatmap(association_rules)
        
        # 10. Transaction Item Distribution
        self._plot_item_frequency(binary_matrix)
        
        # 11. Cumulative Support
        self._plot_cumulative_support(frequent_itemsets)
        
        # 12. Support Percentiles
        self._plot_support_percentiles(frequent_itemsets)
        
        # 13. Association Rule Network (simplified)
        self._plot_rule_network(association_rules)
        
        # 14. Transaction Density
        self._plot_transaction_density(binary_matrix)
        
        # 15. Itemset Co-occurrence Matrix
        self._plot_cooccurrence_matrix(binary_matrix)
        
        logger.info("Visualization generation completed!")
    
    def _plot_support_distribution(self, itemsets: pd.DataFrame):
        """Plot 1: Support distribution histogram"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(itemsets['support'], bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Support', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Itemset Support', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "01_support_distribution.png", dpi=self.dpi)
        plt.close()
        logger.info("Saved: 01_support_distribution.png")
    
    def _plot_support_vs_size(self, itemsets: pd.DataFrame):
        """Plot 2: Support vs itemset size"""
        fig, ax = plt.subplots(figsize=(10, 6))
        itemsets['size'] = itemsets['itemset'].apply(len)
        
        for size in sorted(itemsets['size'].unique()):
            data = itemsets[itemsets['size'] == size]['support']
            ax.scatter([size] * len(data), data, alpha=0.6, s=50, label=f'Size {size}')
        
        ax.set_xlabel('Itemset Size', fontsize=12)
        ax.set_ylabel('Support', fontsize=12)
        ax.set_title('Support vs Itemset Size', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "02_support_vs_size.png", dpi=self.dpi)
        plt.close()
        logger.info("Saved: 02_support_vs_size.png")
    
    def _plot_top_itemsets(self, itemsets: pd.DataFrame, top_k: int = 20):
        """Plot 3: Top-k itemsets by support"""
        fig, ax = plt.subplots(figsize=(12, 6))
        top_items = itemsets.nlargest(top_k, 'support').copy()
        top_items['itemset_str'] = top_items['itemset'].apply(lambda x: str(list(x)[:3]))
        
        ax.barh(range(len(top_items)), top_items['support'], color='steelblue')
        ax.set_yticks(range(len(top_items)))
        ax.set_yticklabels(top_items['itemset_str'])
        ax.set_xlabel('Support', fontsize=12)
        ax.set_title(f'Top-{top_k} Frequent Itemsets', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(self.output_dir / "03_top_itemsets.png", dpi=self.dpi)
        plt.close()
        logger.info("Saved: 03_top_itemsets.png")
    
    def _plot_itemset_size_distribution(self, itemsets: pd.DataFrame):
        """Plot 4: Itemset size distribution"""
        fig, ax = plt.subplots(figsize=(10, 6))
        itemsets['size'] = itemsets['itemset'].apply(len)
        size_dist = itemsets['size'].value_counts().sort_index()
        
        ax.bar(size_dist.index, size_dist.values, color='coral', edgecolor='black')
        ax.set_xlabel('Itemset Size', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Distribution of Itemset Sizes', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(self.output_dir / "04_itemset_size_distribution.png", dpi=self.dpi)
        plt.close()
        logger.info("Saved: 04_itemset_size_distribution.png")
    
    def _plot_confidence_distribution(self, rules: pd.DataFrame):
        """Plot 5: Confidence distribution"""
        if len(rules) == 0:
            logger.warning("No association rules to plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(rules['confidence'], bins=50, edgecolor='black', alpha=0.7, color='green')
        ax.set_xlabel('Confidence', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Rule Confidence', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "05_confidence_distribution.png", dpi=self.dpi)
        plt.close()
        logger.info("Saved: 05_confidence_distribution.png")
    
    def _plot_lift_distribution(self, rules: pd.DataFrame):
        """Plot 6: Lift distribution"""
        if len(rules) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(rules['lift'], bins=50, edgecolor='black', alpha=0.7, color='purple')
        ax.set_xlabel('Lift', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Rule Lift', fontsize=14, fontweight='bold')
        ax.axvline(x=1, color='red', linestyle='--', linewidth=2, label='Lift=1 (No correlation)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "06_lift_distribution.png", dpi=self.dpi)
        plt.close()
        logger.info("Saved: 06_lift_distribution.png")
    
    def _plot_confidence_vs_lift(self, rules: pd.DataFrame):
        """Plot 7: Confidence vs Lift scatter"""
        if len(rules) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(rules['confidence'], rules['lift'], 
                           c=rules['support'], cmap='viridis', s=100, alpha=0.6)
        ax.set_xlabel('Confidence', fontsize=12)
        ax.set_ylabel('Lift', fontsize=12)
        ax.set_title('Confidence vs Lift (colored by Support)', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(scatter)
        cbar.set_label('Support')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "07_confidence_vs_lift.png", dpi=self.dpi)
        plt.close()
        logger.info("Saved: 07_confidence_vs_lift.png")
    
    def _plot_support_vs_confidence(self, rules: pd.DataFrame):
        """Plot 8: Support vs Confidence scatter"""
        if len(rules) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(rules['support'], rules['confidence'],
                           c=rules['lift'], cmap='coolwarm', s=100, alpha=0.6)
        ax.set_xlabel('Support', fontsize=12)
        ax.set_ylabel('Confidence', fontsize=12)
        ax.set_title('Support vs Confidence (colored by Lift)', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(scatter)
        cbar.set_label('Lift')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "08_support_vs_confidence.png", dpi=self.dpi)
        plt.close()
        logger.info("Saved: 08_support_vs_confidence.png")
    
    def _plot_rules_heatmap(self, rules: pd.DataFrame):
        """Plot 9: Association rules heatmap"""
        if len(rules) == 0:
            return
        
        # Create antecedent-consequent matrix
        rules_subset = rules.head(20)  # Top 20 rules
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Simple heatmap of confidence
        confidence_matrix = np.array(rules_subset['confidence']).reshape(-1, 1)
        sns.heatmap(confidence_matrix, cmap='YlOrRd', annot=True, fmt='.2f', ax=ax)
        ax.set_ylabel('Rule Index', fontsize=12)
        ax.set_title('Top 20 Rules - Confidence Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "09_rules_heatmap.png", dpi=self.dpi)
        plt.close()
        logger.info("Saved: 09_rules_heatmap.png")
    
    def _plot_item_frequency(self, binary_matrix: np.ndarray):
        """Plot 10: Item frequency distribution"""
        fig, ax = plt.subplots(figsize=(12, 6))
        item_freq = np.sum(binary_matrix, axis=0)
        
        ax.bar(range(len(item_freq)), item_freq, color='skyblue', edgecolor='black')
        ax.set_xlabel('Item ID', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Item Frequency in Transactions', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(self.output_dir / "10_item_frequency.png", dpi=self.dpi)
        plt.close()
        logger.info("Saved: 10_item_frequency.png")
    
    def _plot_cumulative_support(self, itemsets: pd.DataFrame):
        """Plot 11: Cumulative support distribution"""
        fig, ax = plt.subplots(figsize=(10, 6))
        sorted_support = np.sort(itemsets['support'].values)[::-1]
        cumsum = np.cumsum(sorted_support)
        cumsum_pct = cumsum / cumsum[-1] * 100
        
        ax.plot(cumsum_pct, linewidth=2)
        ax.fill_between(range(len(cumsum_pct)), cumsum_pct, alpha=0.3)
        ax.set_xlabel('Itemset Index (sorted by support)', fontsize=12)
        ax.set_ylabel('Cumulative Support (%)', fontsize=12)
        ax.set_title('Cumulative Support Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "11_cumulative_support.png", dpi=self.dpi)
        plt.close()
        logger.info("Saved: 11_cumulative_support.png")
    
    def _plot_support_percentiles(self, itemsets: pd.DataFrame):
        """Plot 12: Support percentiles"""
        fig, ax = plt.subplots(figsize=(10, 6))
        percentiles = np.percentile(itemsets['support'], np.linspace(0, 100, 11))
        
        ax.bar(range(len(percentiles)), percentiles, color='lightgreen', edgecolor='black')
        ax.set_xlabel('Percentile', fontsize=12)
        ax.set_ylabel('Support', fontsize=12)
        ax.set_title('Support Percentiles', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(percentiles)))
        ax.set_xticklabels([f'{i*10}%' for i in range(11)])
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(self.output_dir / "12_support_percentiles.png", dpi=self.dpi)
        plt.close()
        logger.info("Saved: 12_support_percentiles.png")
    
    def _plot_rule_network(self, rules: pd.DataFrame):
        """Plot 13: Simplified rule network visualization"""
        if len(rules) < 3:
            logger.warning("Not enough rules for network plot")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Simple network-like visualization
        top_rules = rules.nlargest(15, 'lift')
        positions_x = np.random.rand(len(top_rules))
        positions_y = np.random.rand(len(top_rules))
        
        scatter = ax.scatter(positions_x, positions_y, 
                           s=top_rules['confidence']*1000 + 100,
                           c=top_rules['lift'], cmap='RdYlGn', alpha=0.6)
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_title('Rule Network (Size=Confidence, Color=Lift)', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(scatter)
        cbar.set_label('Lift')
        plt.tight_layout()
        plt.savefig(self.output_dir / "13_rule_network.png", dpi=self.dpi)
        plt.close()
        logger.info("Saved: 13_rule_network.png")
    
    def _plot_transaction_density(self, binary_matrix: np.ndarray):
        """Plot 14: Transaction density heatmap"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Sample transactions if too large
        if binary_matrix.shape[0] > 500:
            sample_idx = np.random.choice(binary_matrix.shape[0], 500, replace=False)
            sample_matrix = binary_matrix[sample_idx, :]
        else:
            sample_matrix = binary_matrix
        
        sns.heatmap(sample_matrix[:100, :].astype(int), cmap='Blues', cbar=True, ax=ax)
        ax.set_xlabel('Item ID', fontsize=12)
        ax.set_ylabel('Transaction ID', fontsize=12)
        ax.set_title('Transaction Density Heatmap (First 100 transactions)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "14_transaction_density.png", dpi=self.dpi)
        plt.close()
        logger.info("Saved: 14_transaction_density.png")
    
    def _plot_cooccurrence_matrix(self, binary_matrix: np.ndarray):
        """Plot 15: Item co-occurrence matrix"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Calculate co-occurrence matrix
        cooccurrence = np.dot(binary_matrix.T, binary_matrix)
        
        # Normalize
        cooccurrence_norm = cooccurrence / (np.max(cooccurrence) + 1e-8)
        
        # Limit to top items
        top_items = np.argsort(np.sum(binary_matrix, axis=0))[-20:]
        cooccurrence_subset = cooccurrence_norm[np.ix_(top_items, top_items)]
        
        sns.heatmap(cooccurrence_subset, cmap='YlOrRd', annot=False, ax=ax, cbar_kws={'label': 'Co-occurrence'})
        ax.set_title('Item Co-occurrence Matrix (Top 20 Items)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "15_cooccurrence_matrix.png", dpi=self.dpi)
        plt.close()
        logger.info("Saved: 15_cooccurrence_matrix.png")
