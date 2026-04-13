"""Example usage of CFP-tree library."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cfptree import CFPtree, Visualizer
from cfptree.utils import load_transactions, save_results, generate_report


def example_basic_usage():
    """Basic CFP-tree usage example."""
    print("=" * 60)
    print("EXAMPLE 1: Basic CFP-Tree Usage")
    print("=" * 60)
    
    # Create sample transactions
    transactions = [
        ['a', 'b', 'c'],
        ['a', 'b', 'd'],
        ['a', 'c', 'd'],
        ['b', 'c', 'd'],
        ['a', 'b', 'c', 'd'],
        ['a', 'b'],
        ['b', 'c'],
        ['a', 'c'],
    ]
    
    # Initialize CFP-tree with minimum support of 3
    cfptree = CFPtree(minsup=3)
    
    # Load transactions
    num_transactions = cfptree.read_transactions_from_list(transactions)
    print(f"✓ Loaded {num_transactions} transactions")
    
    # Construct the tree
    cfptree.construct()
    print("✓ CFP-tree constructed")
    
    # Get statistics
    stats = cfptree.get_statistics()
    print(f"\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Get frequent patterns
    patterns = cfptree.get_frequent_patterns()
    print(f"\nFrequent Patterns ({len(patterns)} items):")
    for item, freq in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
        print(f"  {item}: {freq}")
    
    # Get header table
    header_table = cfptree.get_header_table()
    print(f"\nHeader Table:")
    for item, freq in header_table:
        print(f"  {item}: {freq}")
    
    return cfptree


def example_with_file():
    """Example with transaction file."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: CFP-Tree with Transaction File")
    print("=" * 60)
    
    # Path to sample data
    data_file = Path(__file__).parent.parent / 'data' / 'sample_transactions.txt'
    
    if not data_file.exists():
        print(f"⚠ Sample file not found at {data_file}")
        print("Creating sample transactions file...")
        data_file.parent.mkdir(exist_ok=True)
        
        sample_data = [
            "apple banana cherry",
            "apple banana",
            "apple cherry date",
            "banana cherry date",
            "apple banana cherry date",
            "apple banana",
            "banana cherry",
            "apple cherry",
            "cherry date",
            "apple date",
        ]
        
        with open(data_file, 'w') as f:
            f.write('\n'.join(sample_data))
        print(f"✓ Created sample file at {data_file}")
    
    # Initialize and load
    cfptree = CFPtree(minsup=2)
    num_transactions = cfptree.read_transactions(str(data_file))
    print(f"✓ Loaded {num_transactions} transactions from {data_file}")
    
    # Construct
    cfptree.construct()
    print("✓ CFP-tree constructed")
    
    # Display results
    print(f"\nFrequent Items: {len(cfptree.get_frequent_patterns())}")
    print(f"Header Table Size: {len(cfptree.get_header_table())}")
    
    return cfptree


def example_visualization(cfptree: CFPtree):
    """Example visualization."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Visualization")
    print("=" * 60)
    
    visualizer = Visualizer(cfptree)
    
    print("Generating visualizations...")
    output_dir = Path(__file__).parent.parent / 'visualizations'
    visualizer.generate_all_visualizations(str(output_dir))
    print(f"✓ Visualizations saved to {output_dir}")


def example_analysis_report(cfptree: CFPtree):
    """Generate analysis report."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Analysis Report")
    print("=" * 60)
    
    stats = cfptree.get_statistics()
    patterns = cfptree.get_frequent_patterns()
    
    report = generate_report(stats, patterns)
    print(report)
    
    # Save report
    output_file = Path(__file__).parent.parent / 'cfptree_report.txt'
    with open(output_file, 'w') as f:
        f.write(report)
    print(f"✓ Report saved to {output_file}")


if __name__ == '__main__':
    # Run examples
    cfptree1 = example_basic_usage()
    cfptree2 = example_with_file()
    
    # Visualize the file-based example
    example_visualization(cfptree2)
    
    # Generate report
    example_analysis_report(cfptree2)
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)