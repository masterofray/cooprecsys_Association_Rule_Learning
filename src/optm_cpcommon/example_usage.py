"""
Complete example demonstrating the production recommendation system
"""

import logging
import numpy as np
import pandas as pd
from fpgrowth_recommender.pipeline import RecommendationPipeline
from fpgrowth_recommender.mlflow_manager import MLflowManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run complete recommendation pipeline"""
    
    logger.info("Initializing MLflow...")
    MLflowManager.start_mlflow_ui(port=5000)
    
    logger.info("Creating sample e-commerce transaction data...")
    # Generate synthetic e-commerce transactions
    n_transactions = 10000
    n_items = 100
    
    data = np.random.binomial(1, 0.15, size=(n_transactions, n_items))
    df = pd.DataFrame(data, columns=[f'item_{i}' for i in range(n_items)])
    
    # Save to CSV
    df.to_csv("sample_transactions.csv", index=False)
    logger.info(f"Sample data created: {df.shape}")
    
    # Create evaluation data
    eval_predictions = np.random.rand(n_transactions, n_items)
    eval_ground_truth = np.random.binomial(1, 0.1, size=(n_transactions, n_items))
    
    # Run pipeline
    logger.info("Starting recommendation pipeline...")
    pipeline = RecommendationPipeline(config_path="config.ini")
    
    results = pipeline.run_full_pipeline(
        data_source="sample_transactions.csv",
        source_type="csv",
        eval_data=eval_ground_truth.astype(np.float64),
        eval_predictions=eval_predictions
    )
    
    logger.info(f"\n{'='*50}")
    logger.info("Pipeline Results:")
    for key, value in results.items():
        logger.info(f"  {key}: {value}")
    logger.info(f"{'='*50}")
    
    # Generate sample recommendations
    logger.info("\nGenerating sample recommendations...")
    sample_user_items = [1, 5, 10]
    recs = pipeline.recommender.predict(sample_user_items, top_k=5)
    
    logger.info(f"Recommendations for user with items {sample_user_items}:")
    for i, (item, conf, lift) in enumerate(recs, 1):
        logger.info(f"  {i}. Item {item}: confidence={conf:.3f}, lift={lift:.3f}")
    
    logger.info("\nMLflow UI available at: http://localhost:5000")
    logger.info("Visualizations saved to: ./plots/")
    logger.info("Model saved to: ./models/")


if __name__ == "__main__":
    main()