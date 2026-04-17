"""MLflow integration for model tracking and monitoring"""

import logging
import subprocess
import json
from pathlib import Path
import mlflow
import mlflow.sklearn
from datetime import datetime

logger = logging.getLogger(__name__)


class MLflowManager:
    """Manage MLflow experiment tracking"""
    
    def __init__(self, tracking_uri: str = "http://localhost:5000", 
                 experiment_name: str = "fpgrowth_recommender"):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        
        try:
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow connected to {tracking_uri}")
        except Exception as e:
            logger.warning(f"Could not connect to MLflow: {e}")
    
    def start_run(self, run_name: str = None):
        """Start MLflow run"""
        run = mlflow.start_run(run_name=run_name)
        logger.info(f"Started MLflow run: {run.info.run_id}")
        return run
    
    def log_params(self, params: dict):
        """Log parameters"""
        for key, value in params.items():
            mlflow.log_param(key, value)
        logger.info(f"Logged {len(params)} parameters to MLflow")
    
    def log_metrics(self, metrics: dict, step: int = None):
        """Log metrics"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
        logger.info(f"Logged {len(metrics)} metrics to MLflow")
    
    def log_model(self, model, artifact_path: str = "model"):
        """Log model artifact"""
        mlflow.sklearn.log_model(model, artifact_path)
        logger.info(f"Logged model to {artifact_path}")
    
    def log_artifact(self, local_path: str, artifact_path: str = None):
        """Log artifact file"""
        mlflow.log_artifact(local_path, artifact_path)
        logger.info(f"Logged artifact: {local_path}")
    
    def end_run(self):
        """End current MLflow run"""
        mlflow.end_run()
        logger.info("Ended MLflow run")
    
    @staticmethod
    def start_mlflow_ui(port: int = 5000):
        """Start MLflow UI"""
        try:
            subprocess.Popen(['mlflow', 'ui', '--port', str(port)])
            logger.info(f"MLflow UI started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start MLflow UI: {e}")
    
    @staticmethod
    def setup_ngrok(auth_token: str = None):
        """Setup ngrok tunnel for MLflow UI"""
        try:
            from pyngrok import ngrok
            
            if auth_token:
                ngrok.set_auth_token(auth_token)
            
            http_tunnel = ngrok.connect(5000, "http")
            logger.info(f"ngrok tunnel created: {http_tunnel.public_url}")
            return http_tunnel.public_url
        except Exception as e:
            logger.error(f"Failed to setup ngrok: {e}")
            return None