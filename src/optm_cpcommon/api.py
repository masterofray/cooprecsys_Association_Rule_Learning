"""
Production-grade Python API for FPGrowth Recommender System
"""

import logging
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import duckdb
from tqdm import tqdm
from datetime import datetime

from .core.fpgrowth_core import FPGrowthCython
from .core.fpcommon_core import FPCommonCython
from .preprocessing.data_processor import DataProcessorCython
from .metrics.ndcg_metric import NDCGMetricCython
from .visualization.plotter import RecommenderPlotter
from .config import ConfigManager

logger = logging.getLogger(__name__)


class FPGrowthRecommender:
    """
    Main API class for FPGrowth-based recommender system
    
    Attributes
    ----------
    config : ConfigManager
        Configuration manager for all parameters
    model : FPGrowthCython
        Core Cython-based FPGrowth model
    data_processor : DataProcessorCython
        Data preprocessing engine
    frequent_itemsets : pd.DataFrame
        Discovered frequent itemsets
    association_rules : pd.DataFrame
        Generated association rules
    """
    
    def __init__(self, config_path: str = "config.ini"):
        """Initialize recommender system"""
        self.config = ConfigManager(config_path)
        self.model = FPGrowthCython()
        self.data_processor = DataProcessorCython(self.config.get("database", "db_path"))
        self.frequent_itemsets = None
        self.association_rules = None
        self.binary_matrix = None
        self._setup_logging()
        
        logger.info("FPGrowth Recommender initialized")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.config.get("logging", "log_file")
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(log_file)
        handler.setLevel(self.config.get("logging", "log_level"))
        formatter = logging.Formatter(self.config.get("logging", "log_format"))
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    @property
    def min_support(self) -> float:
        """Get minimum support threshold"""
        return self.config.get("model", "min_support")
    
    @property
    def min_confidence(self) -> float:
        """Get minimum confidence threshold"""
        return self.config.get("model", "min_confidence")
    
    def load_data_from_csv(self, csv_path: str, table_name: str = "transactions") -> bool:
        """
        Load transaction data from CSV
        
        Parameters
        ----------
        csv_path : str
            Path to CSV file
        table_name : str
            Table name in DuckDB
        
        Returns
        -------
        bool
            Success status
        """
        logger.info(f"Loading data from {csv_path}")
        return self.data_processor.load_from_csv(csv_path, table_name)
    
    def load_data_from_duckdb(self, query: str, table_name: str = "transactions"):
        """
        Load data from DuckDB query
        
        Parameters
        ----------
        query : str
            SQL query to execute
        table_name : str
            Output table name
        """
        self.data_processor.connect_duckdb()
        df = self.data_processor.connection.execute(query).df()
        logger.info(f"Loaded {len(df)} transactions from DuckDB")
        return df
    
    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess transaction data
        
        Parameters
        ----------
        df : pd.DataFrame
            Input transaction dataframe
        
        Returns
        -------
        np.ndarray
            Binary matrix (n_transactions, n_items)
        """
        logger.info(f"Preprocessing {len(df)} transactions with {len(df.columns)} items")
        FPCommonCython.validate_input(df, null_values=self.config.get("preprocessing", "enable_null_value_handling"))
        
        # Convert to binary matrix
        self.binary_matrix = (df > 0).astype(np.int32).values
        logger.info(f"Binary matrix shape: {self.binary_matrix.shape}")
        return self.binary_matrix
    
    def fit(self, binary_matrix: Optional[np.ndarray] = None):
        """
        Fit FPGrowth model
        
        Parameters
        ----------
        binary_matrix : np.ndarray, optional
            Binary transaction matrix. If None, uses preprocessed data
        """
        if binary_matrix is not None:
            self.binary_matrix = binary_matrix
        
        if self.binary_matrix is None:
            raise ValueError("No data provided. Call preprocess() or pass binary_matrix")
        
        logger.info("Fitting FPGrowth model...")
        min_support = self.min_support
        
        self.model.fit(self.binary_matrix, min_support)
        itemsets, supports = self.model.get_frequent_itemsets()
        
        self.frequent_itemsets = pd.DataFrame({
            'itemset': itemsets,
            'support': supports
        })
        
        logger.info(f"Found {len(self.frequent_itemsets)} frequent itemsets")
    
    def generate_rules(self) -> pd.DataFrame:
        """
        Generate association rules from frequent itemsets
        
        Returns
        -------
        pd.DataFrame
            Association rules with confidence and lift
        """
        if self.frequent_itemsets is None:
            raise ValueError("Model must be fitted first")
        
        logger.info("Generating association rules...")
        min_confidence = self.min_confidence
        min_lift = self.config.get("model", "min_lift")
        
        rules = []
        for _, row in tqdm(self.frequent_itemsets.iterrows(), total=len(self.frequent_itemsets)):
            itemset = row['itemset']
            support = row['support']
            
            if len(itemset) < 2:
                continue
            
            # Generate antecedent-consequent pairs
            for antecedent in self._generate_subsets(itemset):
                consequent = itemset - antecedent
                
                # Calculate confidence and lift
                antecedent_support = self.frequent_itemsets[
                    self.frequent_itemsets['itemset'] == antecedent
                ]['support'].values
                consequent_support = self.frequent_itemsets[
                    self.frequent_itemsets['itemset'] == consequent
                ]['support'].values
                
                if len(antecedent_support) > 0 and len(consequent_support) > 0:
                    confidence = support / antecedent_support[0]
                    lift = support / (antecedent_support[0] * consequent_support[0])
                    
                    if confidence >= min_confidence and lift >= min_lift:
                        rules.append({
                            'antecedent': antecedent,
                            'consequent': consequent,
                            'support': support,
                            'confidence': confidence,
                            'lift': lift
                        })
        
        self.association_rules = pd.DataFrame(rules)
        logger.info(f"Generated {len(self.association_rules)} rules")
        return self.association_rules
    
    @staticmethod
    def _generate_subsets(itemset):
        """Generate non-empty subsets of itemset"""
        from itertools import combinations
        for i in range(1, len(itemset)):
            for subset in combinations(itemset, i):
                yield frozenset(subset)
    
    def predict(self, user_items: List, top_k: int = 10) -> List[Tuple]:
        """
        Generate recommendations for user
        
        Parameters
        ----------
        user_items : List
            Items user has interacted with
        top_k : int
            Number of recommendations to return
        
        Returns
        -------
        List[Tuple]
            Top-k recommendations (item, confidence, lift)
        """
        if self.association_rules is None:
            raise ValueError("Rules must be generated first")
        
        user_itemset = frozenset(user_items)
        recommendations = {}
        
        for _, rule in self.association_rules.iterrows():
            if rule['antecedent'].issubset(user_itemset):
                for item in rule['consequent']:
                    if item not in user_items:
                        if item not in recommendations:
                            recommendations[item] = []
                        recommendations[item].append((rule['confidence'], rule['lift']))
        
        # Aggregate and rank
        final_recommendations = []
        for item, scores in recommendations.items():
            avg_confidence = np.mean([s[0] for s in scores])
            avg_lift = np.mean([s[1] for s in scores])
            final_recommendations.append((item, avg_confidence, avg_lift))
        
        final_recommendations.sort(key=lambda x: (x[1], x[2]), reverse=True)
        logger.info(f"Generated {len(final_recommendations)} recommendations")
        
        return final_recommendations[:top_k]
    
    def evaluate_ndcg(self, ground_truth: np.ndarray, predictions: np.ndarray, k: int = 10) -> Tuple[float, np.ndarray]:
        """
        Evaluate model using NDCG@K metric
        
        Parameters
        ----------
        ground_truth : np.ndarray
            True labels (n_samples, n_items)
        predictions : np.ndarray
            Predicted scores (n_samples, n_items)
        k : int
            Top-k cutoff
        
        Returns
        -------
        Tuple[float, np.ndarray]
            Mean NDCG@K and per-sample scores
        """
        mean_ndcg, scores = NDCGMetricCython.ndcg_at_k(
            predictions.astype(np.float64),
            ground_truth.astype(np.float64),
            k
        )
        logger.info(f"NDCG@{k}: {mean_ndcg:.4f}")
        
        # Store in database
        self._store_ndcg_metric(k, mean_ndcg)
        return mean_ndcg, scores
    
    def _store_ndcg_metric(self, k: int, score: float):
        """Store NDCG metric in database"""
        self.data_processor.connect_duckdb()
        timestamp = datetime.now().isoformat()
        
        query = f"""
        CREATE TABLE IF NOT EXISTS ndcg_metrics (
            timestamp VARCHAR,
            k INTEGER,
            score DOUBLE
        )
        """
        self.data_processor.connection.execute(query)
        
        insert_query = f"INSERT INTO ndcg_metrics VALUES ('{timestamp}', {k}, {score})"
        self.data_processor.connection.execute(insert_query)
        logger.info(f"Stored NDCG@{k} = {score} in database")
    
    def save_model(self, filepath: str):
        """
        Pickle and save model
        
        Parameters
        ----------
        filepath : str
            Path to save model
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'frequent_itemsets': self.frequent_itemsets,
            'association_rules': self.association_rules,
            'binary_matrix': self.binary_matrix,
            'config': self.config.config_dict
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load pickled model
        
        Parameters
        ----------
        filepath : str
            Path to model file
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.frequent_itemsets = model_data['frequent_itemsets']
        self.association_rules = model_data['association_rules']
        self.binary_matrix = model_data['binary_matrix']
        
        logger.info(f"Model loaded from {filepath}")
    
    def visualize(self, output_dir: str = "./plots"):
        """
        Generate all visualizations
        
        Parameters
        ----------
        output_dir : str
            Output directory for plots
        """
        plotter = RecommenderPlotter(output_dir)
        plotter.plot_all(
            frequent_itemsets=self.frequent_itemsets,
            association_rules=self.association_rules,
            binary_matrix=self.binary_matrix
        )