"""Utility functions for CFP-tree."""

import json
from typing import List, Dict, Any, Optional


def load_transactions(filename: str) -> List[List[str]]:
    """Load transactions from file.
    
    Args:
        filename: Path to transaction file (whitespace-separated items per line)
        
    Returns:
        List of transactions
    """
    transactions = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                items = line.strip().split()
                if items:
                    transactions.append(items)
    except FileNotFoundError:
        raise FileNotFoundError(f"Transaction file not found: {filename}")
    
    return transactions


def save_results(results: Dict[str, Any], filename: str) -> None:
    """Save analysis results to JSON.
    
    Args:
        results: Dictionary of results
        filename: Output filename
    """
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)


def load_results(filename: str) -> Dict[str, Any]:
    """Load analysis results from JSON.
    
    Args:
        filename: Input filename
        
    Returns:
        Dictionary of results
    """
    with open(filename, 'r') as f:
        return json.load(f)


def format_results(patterns: Dict[str, int], min_freq: int = 1) -> List[tuple]:
    """Format pattern results.
    
    Args:
        patterns: Pattern dictionary
        min_freq: Minimum frequency filter
        
    Returns:
        Sorted list of (pattern, frequency) tuples
    """
    filtered = [(item, freq) for item, freq in patterns.items() if freq >= min_freq]
    return sorted(filtered, key=lambda x: x[1], reverse=True)


def generate_report(stats: Dict[str, Any], patterns: Dict[str, int]) -> str:
    """Generate text report.
    
    Args:
        stats: Statistics dictionary
        patterns: Pattern dictionary
        
    Returns:
        Formatted report string
    """
    report = (
        "=" * 60 + "\n"
        "CFP-TREE ANALYSIS REPORT\n"
        "=" * 60 + "\n\n"
        f"STATISTICS:\n"
        f"  Minimum Support: {stats['min_support']}\n"
        f"  Total Transactions: {stats['total_transactions']}\n"
        f"  Unique Items: {stats['unique_items']}\n"
        f"  Frequent Items: {stats['frequent_items']}\n"
        f"  Avg Transaction Size: {stats['avg_transaction_size']:.2f}\n"
        f"  Frequent Item Ratio: {stats['frequent_items']/max(stats['unique_items'], 1)*100:.2f}%\n\n"
        f"TOP 10 PATTERNS:\n"
    )
    
    sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
    for i, (item, freq) in enumerate(sorted_patterns[:10], 1):
        report += f"  {i}. {item}: {freq}\n"
    
    report += "=" * 60 + "\n"
    return report