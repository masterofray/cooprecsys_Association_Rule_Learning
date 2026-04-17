from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np
import os

extensions = [
    "fpgrowth_recommender/core/fpgrowth_core.pyx",
    "fpgrowth_recommender/core/fpcommon_core.pyx",
    "fpgrowth_recommender/preprocessing/data_processor.pyx",
    "fpgrowth_recommender/metrics/ndcg_metric.pyx",
]

setup(
    name="fpgrowth_recommender",
    version="1.0.0",
    author="Your Name",
    description="Production-ready FPGrowth-based recommender system with Cython optimization",
    packages=find_packages(),
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3", "boundscheck": False, "wraparound": False},
        annotate=True,
    ),
    include_dirs=[np.get_include()],
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "duckdb>=0.5.0",
        "mlflow>=1.20.0",
        "tqdm>=4.62.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "Cython>=0.29.0",
    ],
)