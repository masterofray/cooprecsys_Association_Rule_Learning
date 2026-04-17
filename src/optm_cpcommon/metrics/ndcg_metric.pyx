# distutils: language = c
# cython: language_level=3, boundscheck=False, wraparound=False

from libc.stdio cimport fprintf, stdout
from libc.math cimport log2
import numpy as np
cimport numpy as np
from cython.parallel cimport prange

ctypedef np.int32_t DTYPE_int
ctypedef np.float64_t DTYPE_float


cdef class NDCGMetricCython:
    """NDCG@K metric implementation in Cython"""
    
    @staticmethod
    cdef double dcg_at_k(np.ndarray[DTYPE_float, ndim=1] relevances, int k):
        """Calculate DCG@K"""
        cdef double dcg = 0.0
        cdef int i
        cdef int n = min(k, len(relevances))
        
        for i in range(n):
            if relevances[i] > 0:
                dcg += relevances[i] / log2(<double>(i + 2))
        
        return dcg
    
    @staticmethod
    def ndcg_at_k(np.ndarray[DTYPE_float, ndim=2] recommendations, 
                  np.ndarray[DTYPE_float, ndim=2] ground_truth, int k):
        """
        Calculate NDCG@K metric
        
        Parameters
        ----------
        recommendations : np.ndarray (n_samples, n_items)
            Predicted scores/rankings
        ground_truth : np.ndarray (n_samples, n_items)
            True labels (1 for relevant, 0 for not relevant)
        k : int
            Top-k cutoff
        
        Returns
        -------
        float
            Mean NDCG@K score
        """
        cdef int n_samples = recommendations.shape[0]
        cdef int n_items = recommendations.shape[1]
        cdef np.ndarray[DTYPE_float, ndim=1] ndcg_scores = np.zeros(n_samples, dtype=np.float64)
        cdef int i, j
        cdef np.ndarray[DTYPE_float, ndim=1] rel
        cdef np.ndarray[np.intp_t, ndim=1] top_k_indices
        cdef np.ndarray[DTYPE_float, ndim=1] ideal_rel
        cdef np.ndarray[np.intp_t, ndim=1] ideal_indices
        cdef double dcg, idcg
        
        fprintf(stdout, "NDCGMetric: Calculating NDCG@%d for %d samples\n", k, n_samples)
        
        for i in prange(n_samples, nogil=False, schedule='dynamic'):
            # Get top-k recommendations
            top_k_indices = np.argsort(-recommendations[i, :])[:k]
            rel = ground_truth[i, top_k_indices]
            dcg = NDCGMetricCython.dcg_at_k(rel, k)
            
            # Get ideal ranking
            ideal_rel = np.sort(-ground_truth[i, :])[:k]
            ideal_rel = -ideal_rel
            idcg = NDCGMetricCython.dcg_at_k(ideal_rel, k)
            
            if idcg > 0:
                ndcg_scores[i] = dcg / idcg
            else:
                ndcg_scores[i] = 0.0
        
        cdef double mean_ndcg = np.mean(ndcg_scores)
        fprintf(stdout, "NDCGMetric: Mean NDCG@%d = %.4f\n", k, mean_ndcg)
        
        return mean_ndcg, ndcg_scores