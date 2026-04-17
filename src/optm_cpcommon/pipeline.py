"""
Production pipeline orchestration
"""

import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from .api import FPGrowthRecommender
from .mlflow_manager import MLflowManager

logger = logging.getLogger(__name__)


class RecommendationPipeline:
    """Complete production recommendation pipeline"""
    
    def __init__(self, config_path: str = "config.ini"):
        self.config_path = config_path
        self.recommender = FPGrowthRecommender(config_path)
        self.mlflow = MLflowManager()
        self.pipeline_start_time = None
    
    def run_full_pipeline(self, data_source: str, source_type: str = "csv", 
                         eval_data: Optional[np.ndarray] = None,
                         eval_predictions: Optional[np.ndarray] = None) -> dict:
        """
        Run complete recommendation pipeline
        
        Parameters
        ----------
        data_source : str
            Path to data or SQL query
        source_type : str
            Type of data source ('csv', 'duckdb_query', 'dataframe')
        eval_data : np.ndarray, optional
            Ground truth for evaluation
        eval_predictions : np.ndarray, optional
            Predictions for evaluation
        
        Returns
        -------
        dict
            Pipeline results and metrics
        """
        self.pipeline_start_time = datetime.now()
        logger.info("=" * 50)
        logger.info("Starting Recommendation Pipeline")
        logger.info("=" * 50)
        
        results = {}
        
        # Step 1: Load Data
        logger.info("\n[Step 1] Loading Data...")
        df = self._load_data(data_source, source_type)
        results['n_transactions'] = len(df)
        results['n_items'] = len(df.columns)
        
        # Step 2: Preprocess
        logger.info("\n[Step 2] Preprocessing...")
        binary_matrix = self.recommender.preprocess(df)
        results['sparsity'] = 1.0 - (np.count_nonzero(binary_matrix) / binary_matrix.size)
        
        # Step 3: Model Training with MLflow tracking
        logger.info("\n[Step 3] Training Model...")
        with self.mlflow.start_run(run_name=f"fpgrowth_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            self._log_config_params()
            self.recommender.fit(binary_matrix)
            results['n_itemsets'] = len(self.recommender.frequent_itemsets)
        
        # Step 4: Generate Rules
        logger.info("\n[Step 4] Generating Association Rules...")
        rules = self.recommender.generate_rules()
        results['n_rules'] = len(rules)
        
        if len(rules) > 0:
            results['mean_confidence'] = rules['confidence'].mean()
            results['mean_lift'] = rules['lift'].mean()
        
        # Step 5: Evaluation (if provided)
        if eval_data is not None and eval_predictions is not None:
            logger.info("\n[Step 5] Evaluation...")
            ndcg_score, _ = self.recommender.evaluate_ndcg(eval_data, eval_predictions, k=10)
            results['ndcg@10'] = ndcg_score
        
        # Step 6: Visualization
        logger.info("\n[Step 6] Generating Visualizations...")
        self.recommender.visualize()
        
        # Step 7: Save Model
        logger.info("\n[Step 7] Saving Model...")
        model_path = f"./models/fpgrowth_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        self.recommender.save_model(model_path)
        results['model_path'] = model_path
        
        # Log metrics to MLflow
        self.mlflow.log_metrics(results)
        self.mlflow.end_run()
        
        # Summary
        pipeline_duration = (datetime.now() - self.pipeline_start_time).total_seconds()
        logger.info("\n" + "=" * 50)
        logger.info("Pipeline Completed Successfully!")
        logger.info("=" * 50)
        logger.info(f"Duration: {pipeline_duration:.2f}s")
        logger.info(f"Results:\n{json.dumps(results, indent=2, default=str)}")
        
        return results
    
    def _load_data(self, source: str, source_type: str) -> pd.DataFrame:
        """Load data from various sources"""
        if source_type == "csv":
            logger.info(f"Loading CSV from {source}")
            return pd.read_csv(source)
        elif source_type == "duckdb_query":
            logger.info(f"Loading from DuckDB query: {source}")
            return self.recommender.load_data_from_duckdb(source)
        else:
            raise ValueError(f"Unknown source type: {source_type}")
    
    def _log_config_params(self):
        """Log configuration parameters to MLflow"""
        params = {
            'min_support': self.recommender.min_support,
            'min_confidence': self.recommender.min_confidence,
            'enable_parallel': self.recommender.config.get('performance', 'enable_parallel_processing'),
            'num_workers': self.recommender.config.get('performance', 'num_workers'),
        }
        self.mlflow.log_params(params)
    
    def batch_predict(self, user_histories: list, top_k: int = 10) -> list:
        """
        Generate recommendations for batch of users
        
        Parameters
        ----------
        user_histories : list
            List of user item histories
        top_k : int
            Number of recommendations per user
        
        Returns
        -------
        list
            Recommendations for each user
        """
        logger.info(f"Generating recommendations for {len(user_histories)} users...")
        
        all_recommendations = []
        for i, user_items in enumerate(tqdm(user_histories)):
            try:
                recs = self.recommender.predict(user_items, top_k)
                all_recommendations.append(recs)
            except Exception as e:
                logger.error(f"Error generating recommendations for user {i}: {e}")
                all_recommendations.append([])
        
        return all_recommendations