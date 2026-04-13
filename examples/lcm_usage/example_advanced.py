"""
Advanced examples showing LCM capabilities.
"""

from lcm import LCM, LCMRunner
from lcm.utils import (
    filter_by_support, 
    filter_by_size,
    get_maximal_itemsets,
    get_closed_itemsets,
    association_rules_from_itemsets,
    itemset_to_string
)
import time

# Example Dataset
transactions = [
    [1, 2, 3, 4],
    [1, 2, 3],
    [1, 2, 4],
    [2, 3, 4],
    [1, 3, 4],
    [1, 2, 3, 4, 5],
    [2, 3, 5],
    [1, 5],
    [3, 4, 5],
    [1, 2, 5],
]

print("=" * 70)
print("ADVANCED LCM EXAMPLES")
print("=" * 70)

# Example 1: Performance Comparison
print("\n1. PERFORMANCE COMPARISON")
print("-" * 70)

min_supports = [2, 3, 4, 5]

for min_sup in min_supports:
    start = time.time()
    itemsets = LCM.mine_closed_itemsets(transactions, min_support=min_sup)
    elapsed = time.time() - start
    
    print(f"Min Support: {min_sup}")
    print(f"  Found: {len(itemsets)} closed itemsets")
    print(f"  Time: {elapsed*1000:.2f}ms")

# Example 2: Filtering Results
print("\n2. FILTERING RESULTS")
print("-" * 70)

itemsets = LCM.mine_closed_itemsets(transactions, min_support=2)
print(f"All closed itemsets (support >= 2): {len(itemsets)}")

# Filter by size
size_filtered = filter_by_size(itemsets, min_size=2, max_size=3)
print(f"Filtered (size 2-3): {len(size_filtered)}")
for items, sup in size_filtered:
    print(f"  {set(items)}: support={sup}")

# Example 3: Extraction of Special Itemsets
print("\n3. EXTRACTION OF SPECIAL ITEMSETS")
print("-" * 70)

frequent = LCM.mine_frequent_itemsets(transactions, min_support=2)
print(f"Frequent itemsets: {len(frequent)}")

maximal = get_maximal_itemsets(frequent)
print(f"Maximal itemsets: {len(maximal)}")
for items, sup in maximal:
    print(f"  {set(items)}: support={sup}")

closed = get_closed_itemsets(frequent)
print(f"Closed itemsets: {len(closed)}")

# Example 4: Association Rules with Different Confidence Levels
print("\n4. ASSOCIATION RULES")
print("-" * 70)

itemsets = LCM.mine_closed_itemsets(transactions, min_support=2)

for min_conf in [0.5, 0.6, 0.7]:
    rules = association_rules_from_itemsets(itemsets, min_confidence=min_conf)
    print(f"\nRules with confidence >= {min_conf}: {len(rules)}")
    for rule in rules[:5]:  # Show top 5
        print(f"  {rule['antecedent']} => {rule['consequent']}")
        print(f"    Confidence: {rule['confidence']:.3f}, Support: {rule['support']}")

# Example 5: Custom Runner with Options
print("\n5. CUSTOM RUNNER WITH OPTIONS")
print("-" * 70)

runner = LCMRunner(mode='closed', min_support=2, verbose=True)
runner.load_transactions(transactions)
results = runner.mine()

print(f"\nMined {len(results)} itemsets:")
for itemset, support in sorted(results, key=lambda x: (-x[1], x[0]))[:10]:
    print(f"  {set(itemset)}: {support}")

# Example 6: String Representations
print("\n6. STRING REPRESENTATIONS")
print("-" * 70)

itemsets = LCM.mine_closed_itemsets(transactions, min_support=3)
for items, sup in itemsets:
    string_repr = itemset_to_string(items, separator='-')
    print(f"  Items: {string_repr}, Support: {sup}")

# Example 7: Mining Different Problem Types
print("\n7. DIFFERENT MINING MODES")
print("-" * 70)

min_sup = 2

frequent = LCM.mine_frequent_itemsets(transactions, min_support=min_sup)
closed = LCM.mine_closed_itemsets(transactions, min_support=min_sup)
maximal = LCM.mine_maximal_itemsets(transactions, min_support=min_sup)

print(f"Frequent itemsets:  {len(frequent)}")
print(f"Closed itemsets:    {len(closed)}")
print(f"Maximal itemsets:   {len(maximal)}")

print("\n" + "=" * 70)
print("Advanced examples completed!")
print("=" * 70)