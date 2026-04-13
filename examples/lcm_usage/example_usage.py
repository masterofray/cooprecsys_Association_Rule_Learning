"""
Example usage of LCM itemset mining library.
"""

from lcm import LCM, LCMRunner

# Example 1: Mine closed frequent itemsets
print("=" * 50)
print("Example 1: Closed Itemset Mining")
print("=" * 50)

transactions = [
    [1, 2, 3],
    [1, 2],
    [1, 3],
    [2, 3],
    [1, 2, 3, 4],
]

closed_itemsets = LCM.mine_closed_itemsets(transactions, min_support=2)
print(f"Closed itemsets (min_support=2):")
for itemset, support in sorted(closed_itemsets, key=lambda x: (-x[1], x[0])):
    print(f"  {set(itemset)}: support={support}")

# Example 2: Mine frequent itemsets
print("\n" + "=" * 50)
print("Example 2: Frequent Itemset Mining")
print("=" * 50)

frequent_itemsets = LCM.mine_frequent_itemsets(transactions, min_support=2)
print(f"Frequent itemsets (min_support=2):")
for itemset, support in sorted(frequent_itemsets, key=lambda x: (-x[1], x[0])):
    print(f"  {set(itemset)}: support={support}")

# Example 3: Mine maximal itemsets
print("\n" + "=" * 50)
print("Example 3: Maximal Itemset Mining")
print("=" * 50)

maximal_itemsets = LCM.mine_maximal_itemsets(transactions, min_support=2)
print(f"Maximal itemsets (min_support=2):")
for itemset, support in sorted(maximal_itemsets, key=lambda x: (-x[1], x[0])):
    print(f"  {set(itemset)}: support={support}")

# Example 4: Using LCMRunner for more control
print("\n" + "=" * 50)
print("Example 4: Using LCMRunner")
print("=" * 50)

runner = LCMRunner(mode='closed', min_support=2)
runner.load_transactions(transactions)
results = runner.mine()
print(f"Results from LCMRunner: {len(results)} itemsets found")

# Example 5: Generate association rules
print("\n" + "=" * 50)
print("Example 5: Association Rules")
print("=" * 50)

from lcm.utils import association_rules_from_itemsets

rules = association_rules_from_itemsets(closed_itemsets, min_confidence=0.5)
print(f"Association rules (min_confidence=0.5):")
for rule in rules:
    print(f"  {rule['antecedent']} -> {rule['consequent']}: "
          f"confidence={rule['confidence']:.2f}, support={rule['support']}")

print("\n" + "=" * 50)
print("All examples completed!")
print("=" * 50)