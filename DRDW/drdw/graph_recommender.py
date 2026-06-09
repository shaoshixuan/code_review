from __future__ import print_function
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse import hstack, vstack


class GraphRec(object):
    """
    Bipartite graph recommender using per-user random walk propagation.
    Instead of computing the full P^m matrix (very expensive), we do
    per-user vector propagation which is fast (0.02s per user).
    """

    def __init__(self, train_matrix):
        if not isinstance(train_matrix, csr_matrix):
            self.train_matrix = csr_matrix(train_matrix).astype(np.float32)
        else:
            self.train_matrix = train_matrix.astype(np.float32)

        self.num_u = self.train_matrix.shape[0]
        self.num_i = self.train_matrix.shape[1]

        # Build bipartite adjacency matrix
        block1 = csr_matrix((self.num_u, self.num_u), dtype=np.float32)
        block2 = csr_matrix((self.num_i, self.num_i), dtype=np.float32)
        upper = hstack([block1, self.train_matrix])
        lower = hstack([self.train_matrix.T, block2])
        self.A = vstack([upper, lower])

        # Degree-normalized transition matrix
        self.D = np.array(self.A.sum(axis=1)).flatten()
        self.D[self.D == 0] = 0.0001
        self.P = self.A.multiply(1.0 / self.D[:, None])
        self.P_multi = {}
        self.P_multi[1] = self.P

        # Cache for per-user results (user_id, hops) -> dense item scores
        self._user_cache = {}
        self._current_user = None

    def set_current_user(self, user_id):
        """Prepare per-user computation."""
        self._current_user = user_id

    def performMultiHop(self, m):
        """
        For the current user, compute m-hop random walk scores via
        per-user vector propagation (much faster than full matrix multiply).

        Returns a fake 2D "matrix" accessor compatible with the original
        Sample_And_Rank code:  prob[user_id, start_col:]
        We return a PerUserProxy object.
        """
        if self._current_user is None:
            raise RuntimeError("Call set_current_user() before performMultiHop().")

        cache_key = (self._current_user, m)
        if cache_key in self._user_cache:
            return _PerUserProxy(self._user_cache[cache_key], self._current_user, self.num_u)

        # Per-user vector propagation
        n_total = self.num_u + self.num_i
        user_vec = np.zeros((1, n_total), dtype=np.float32)
        user_vec[0, self._current_user] = 1.0
        pv = csr_matrix(user_vec)
        for _ in range(m):
            pv = pv.dot(self.P)

        scores = pv.toarray()[0]  # shape (n_total,)
        self._user_cache[cache_key] = scores

        return _PerUserProxy(scores, self._current_user, self.num_u)

    def clear_user_cache(self):
        self._user_cache = {}


class _PerUserProxy:
    """
    A lightweight proxy so that the original Sample_And_Rank code can do:
        prob = self.Model_RDW.performMultiHop(currentHop)
        recs = prob[user_id, start_col:]
        recs_dense = recs.toarray().flatten()
    """
    def __init__(self, scores, user_id, num_u):
        self._scores = scores   # shape (num_u + num_i,)
        self._user_id = user_id
        self._num_u = num_u

    def __getitem__(self, key):
        row_idx, col_slice = key
        assert row_idx == self._user_id, \
            f"PerUserProxy: expected user {self._user_id}, got {row_idx}"
        sliced = self._scores[col_slice]
        return _ArrayProxy(sliced)


class _ArrayProxy:
    """Wrap a 1D numpy array so .toarray().flatten() works."""
    def __init__(self, arr):
        self._arr = arr

    def toarray(self):
        return self._arr.reshape(1, -1)
