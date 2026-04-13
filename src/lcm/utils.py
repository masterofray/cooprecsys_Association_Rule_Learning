"""
Utility functions for LCM mining.
"""

def itemset_to_string(itemset, separator=','):
    """Convert itemset to string representation."""
    return separator.join(str(item) for item in sorted(itemset))

def filter_by_support(itemsets, min_support):
    """Filter itemsets by minimum support."""
    return [(items, sup) for items, sup in itemsets if sup >= min_support]

def filter_by_size(itemsets, min_size=None, max_size=None):
    """Filter itemsets by size."""
    result = []
    for items, sup in itemsets:
        size = len(items)
        if (min_size is None or size >= min_size) and \
           (max_size is None or size <= max_size):
            result.append((items, sup))
    return result

def get_maximal_itemsets(itemsets):
    """Get maximal itemsets from frequent itemsets."""
    itemset_list = [set(items) for items, _ in itemsets]
    maximal = []
    
    for i, itemset in enumerate(itemset_list):
        is_maximal = True
        for j, other in enumerate(itemset_list):
            if i != j and itemset < other:
                is_maximal = False
                break
        if is_maximal:
            maximal.append(itemsets[i])
    
    return maximal

def get_closed_itemsets(itemsets):
    """Get closed itemsets from frequent itemsets."""
    itemset_list = [(set(items), sup) for items, sup in itemsets]
    closed = []
    
    for itemset, sup in itemset_list:
        is_closed = True
        for other_itemset, other_sup in itemset_list:
            if itemset < other_itemset and sup == other_sup:
                is_closed = False
                break
        if is_closed:
            closed.append((list(itemset), sup))
    
    return closed

def association_rules_from_itemsets(itemsets, min_confidence=0.5):
    """Generate association rules from itemsets."""
    rules = []
    
    for items, support in itemsets:
        if len(items) < 2:
            continue
        
        items_set = set(items)
        # Generate all possible rules X -> Y
        for i in range(1, len(items)):
            from itertools import combinations
            for antecedent in combinations(items, i):
                antecedent_set = set(antecedent)
                consequent_set = items_set - antecedent_set
                
                # Find support of antecedent
                antecedent_support = None
                for candidate_items, candidate_sup in itemsets:
                    if set(candidate_items) == antecedent_set:
                        antecedent_support = candidate_sup
                        break
                
                if antecedent_support:
                    confidence = support / antecedent_support
                    if confidence >= min_confidence:
                        rules.append({
                            'antecedent': list(antecedent_set),
                            'consequent': list(consequent_set),
                            'support': support,
                            'confidence': confidence,
                            'lift': support / (antecedent_support * len(itemsets)) if itemsets else 0
                        })
    
    return rules