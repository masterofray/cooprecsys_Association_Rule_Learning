# distutils: language = c
# cython: language_level=3, boundscheck=False, wraparound=False

from libc.stdio cimport fprintf, stdout
import numpy as np
cimport numpy as np
from cython.parallel cimport prange
from libc.omp cimport omp_get_num_threads
import duckdb
import logging

ctypedef np.int32_t DTYPE_int
ctypedef np.float64_t DTYPE_float

logger = logging.getLogger(__name__)


cdef class DataProcessorCython:
    """High-performance data preprocessing in Cython"""
    
    cdef public object db_path
    cdef public object connection
    
    def __init__(self, str db_path="./data/recommender.db"):
        self.db_path = db_path
        self.connection = None
    
    def connect_duckdb(self):
        """Connect to DuckDB"""
        try:
            self.connection = duckdb.connect(self.db_path)
            fprintf(stdout, "DataProcessor: Connected to DuckDB at %s\n", 
                    self.db_path.encode('utf-8'))
            return True
        except Exception as e:
            logger.error(f"Failed to connect to DuckDB: {e}")
            return False
    
    def load_from_csv(self, str csv_path, str table_name):
        """Load CSV data into DuckDB"""
        if self.connection is None:
            self.connect_duckdb()
        
        try:
            query = f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{csv_path}')"
            self.connection.execute(query)
            fprintf(stdout, "DataProcessor: Loaded CSV %s into table %s\n", 
                    csv_path.encode('utf-8'), table_name.encode('utf-8'))
            return True
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            return False
    
    def preprocess_binary_matrix(self, str table_name, list item_cols, 
                                  list transaction_cols, double min_support=0.01):
        """Convert transaction data to binary matrix"""
        query = f"""
        SELECT {', '.join(transaction_cols)}, {', '.join(item_cols)}
        FROM {table_name}
        """
        
        cdef object df = self.connection.execute(query).df()
        cdef int n_rows = len(df)
        cdef int n_items = len(item_cols)
        cdef np.ndarray[DTYPE_int, ndim=2] binary_matrix = np.zeros((n_rows, n_items), dtype=np.int32)
        
        fprintf(stdout, "DataProcessor: Converting to binary matrix (%d x %d)\n", n_rows, n_items)
        
        # Parallel conversion to binary matrix
        cdef int i, j
        for i in prange(n_rows, nogil=True, schedule='dynamic'):
            for j in range(n_items):
                if df[item_cols[j]].iloc[i] > 0:
                    binary_matrix[i, j] = 1
        
        fprintf(stdout, "DataProcessor: Binary matrix created successfully\n")
        return binary_matrix
    
    def store_results(self, str table_name, object itemsets_df):
        """Store itemsets results in DuckDB"""
        if self.connection is None:
            self.connect_duckdb()
        
        try:
            self.connection.register(table_name, itemsets_df)
            self.connection.execute(f"CREATE TABLE {table_name}_results AS SELECT * FROM {table_name}")
            fprintf(stdout, "DataProcessor: Stored results in table %s_results\n", 
                    table_name.encode('utf-8'))
            return True
        except Exception as e:
            logger.error(f"Failed to store results: {e}")
            return False