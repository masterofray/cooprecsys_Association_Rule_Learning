"""
Example showing how to work with itemsets from files.
"""

import csv
import os
from lcm import LCM
from lcm.utils import association_rules_from_itemsets

def load_transactions_from_csv(filename):
    """Load transactions from CSV file."""
    transactions = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header if exists
        for row in reader:
            # Convert to integers, filter empty
            itemset = [int(x.strip()) for x in row if x.strip()]
            if itemset:
                transactions.append(itemset)
    return transactions

def save_itemsets_to_csv(itemsets, filename):
    """Save itemsets to CSV file."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Itemset', 'Support'])
        for items, support in itemsets:
            writer.writerow([','.join(map(str, items)), support])

def save_rules_to_csv(rules, filename):
    """Save association rules to CSV file."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Antecedent', 'Consequent', 'Support', 'Confidence', 'Lift'])
        for rule in rules:
            antecedent = ','.join(map(str, rule['antecedent']))
            consequent = ','.join(map(str, rule['consequent']))
            writer.writerow([
                antecedent,
                consequent,
                rule['support'],
                f"{rule['confidence']:.4f}",
                f"{rule['lift']:.4f}"
            ])

# Example usage
if __name__ == "__main__":
    print("File-based LCM Example")
    print("=" * 50)
    
    # Create sample input file
    sample_data = """item1,item2,item3,item4
1,2,3,4
1,2,3
1,2,4
2,3,4
1,3,4
1,2,3,4,5
2,3,5
1,5
3,4,5
1,2,5
"""
    
    input_file = "transactions.csv"
    with open(input_file, 'w') as f:
        f.write(sample_data)
    
    # Load transactions
    transactions = load_transactions_from_csv(input_file)
    print(f"Loaded {len(transactions)} transactions")
    
    # Mine itemsets
    itemsets = LCM.mine_closed_itemsets(transactions, min_support=2)
    print(f"Found {len(itemsets)} closed itemsets")
    
    # Save itemsets
    itemsets_file = "itemsets.csv"
    save_itemsets_to_csv(itemsets, itemsets_file)
    print(f"Saved itemsets to {itemsets_file}")
    
    # Generate rules
    rules = association_rules_from_itemsets(itemsets, min_confidence=0.5)
    print(f"Found {len(rules)} association rules")
    
    # Save rules
    rules_file = "rules.csv"
    save_rules_to_csv(rules, rules_file)
    print(f"Saved rules to {rules_file}")
    
    # Cleanup
    for f in [input_file, itemsets_file, rules_file]:
        if os.path.exists(f):
            os.remove(f)
    
    print("\nExample completed successfully!")