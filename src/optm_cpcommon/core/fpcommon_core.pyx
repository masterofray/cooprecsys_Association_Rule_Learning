# distutils: language = c
# cython: language_level=3, boundscheck=False, wraparound=False

from libc.stdio cimport fprintf, stdout
import numpy as np
cimport numpy as np
from cpython cimport PyObject
import pandas as pd

ctypedef np.int32_t DTYPE_int
ctypedef np.float64_t DTYPE_float

cdef extern from "stdio.h":
    int printf(const char* format, ...)


cdef class FPCommonCython:
    """Common utilities for FP-tree based algorithms"""
    
    @staticmethod
    def validate_input(object df, bint null_values=False):
        """Validate and preprocess input data"""
        if df is None:
            return
        
        if df.size == 0:
            return
        
        # Check for non-boolean types
        cdef bint all_bools = True
        if null_values:
            all_bools = (
                df.apply(lambda col: col.apply(lambda x: pd.isna(x) or isinstance(x, bool)))
                .all()
                .all()
            )
        else:
            all_bools = df.dtypes.apply(pd.api.types.is_bool_dtype).all()
        
        if not all_bools:
            import warnings
            warnings.warn(
                "DataFrames with non-bool types result in worse computational performance",
                DeprecationWarning,
            )
        
        fprintf(stdout, "FPCommon: Input validation passed\n")
    
    @staticmethod
    def setup_fptree(object df, double min_support, bint null_values=False):
        """Setup FP-tree from DataFrame"""
        cdef long num_transactions = len(df.index)
        cdef int num_items = len(df.columns)
        
        fprintf(stdout, "FPCommon: Setting up FP-tree with %ld transactions and %d items\n", 
                num_transactions, num_items)
        
        # Calculate item support
        cdef np.ndarray item_support
        if null_values:
            disabled = df.copy()
            disabled = np.where(pd.isna(disabled), 1, np.nan) + np.where(
                (disabled == 0) | (disabled == 1), np.nan, 0
            )
            item_support = np.array(
                np.nansum(df.values, axis=0)
                / (float(num_transactions) - np.nansum(disabled, axis=0))
            )
        else:
            item_support = np.array(np.sum(df.values, axis=0) / float(num_transactions))
        
        item_support = item_support.reshape(-1)
        cdef np.ndarray items = np.nonzero(item_support >= min_support)[0]
        
        fprintf(stdout, "FPCommon: Found %d frequent items\n", len(items))
        
        # Define ordering
        cdef np.ndarray indices = item_support[items].argsort()
        cdef dict rank = {int(item): i for i, item in enumerate(items[indices])}
        
        return rank, item_support, disabled if null_values else None
    
    @staticmethod
    def generate_itemsets(object generator, object df, object disabled, 
                         double min_support, long num_itemsets, 
                         dict colname_map=None, bint null_values=False):
        """Generate itemsets from FPGrowth generator"""
        cdef list itemsets = []
        cdef list supports = []
        
        fprintf(stdout, "FPCommon: Generating itemsets\n")
        
        if not null_values or disabled is None:
            for sup, iset in generator:
                support = sup / <double>num_itemsets
                if support >= min_support:
                    itemsets.append(frozenset(iset))
                    supports.append(support)
        else:
            for sup, iset in generator:
                itemsets.append(frozenset(iset))
                dec = disabled[:, iset]
                _dec = df.values[:, iset]
                
                if len(iset) == 1:
                    supports.append((sup - np.nansum(dec)) / (num_itemsets - np.nansum(dec)))
                elif len(iset) > 1:
                    denom = 0
                    num = 0
                    for i in range(dec.shape[0]):
                        item_dsbl = list(dec[i, :])
                        item_orig = list(_dec[i, :])
                        
                        if 1 in set(item_dsbl):
                            denom += 1
                            if (0 not in set(item_orig)) or (all(np.isnan(x) for x in item_orig)):
                                num -= 1
                    
                    if num_itemsets - denom == 0:
                        supports.append(0)
                    else:
                        supports.append((sup + num) / (num_itemsets - denom))
        
        res_df = pd.DataFrame({"support": supports, "itemsets": itemsets})
        res_df = res_df[res_df["support"] >= min_support]
        
        if colname_map is not None:
            res_df["itemsets"] = res_df["itemsets"].apply(
                lambda x: frozenset([colname_map[i] for i in x])
            )
        
        fprintf(stdout, "FPCommon: Generated %ld itemsets\n", len(res_df))
        
        return res_df