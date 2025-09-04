# src/debiasing/IFWalk.py
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from gensim.models import Word2Vec

def _to_float32(x):
    x = np.asarray(x)
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return x

def _row_normalize_csr(S: sp.csr_matrix) -> sp.csr_matrix:
    S = S.tocsr(copy=True)
    rs = np.array(S.sum(1)).ravel()
    rs[rs == 0.0] = 1.0
    inv = 1.0 / rs
    # scale rows in-place
    for i in range(S.shape[0]):
        start, end = S.indptr[i], S.indptr[i + 1]
        if start == end:
            continue
        S.data[start:end] *= inv[i]
    return S

def build_similarity_knn(
    X: np.ndarray,
    topk: int = 20,
    metric: str = "cosine",
    n_jobs: int = -1,
) -> sp.csr_matrix:
    """
    Memory-safe kNN similarity graph:
      - No dense n×n matrix.
      - Returns row-normalized CSR similarity.
    """
    X = _to_float32(X)
    n = X.shape[0]
    # +1 to include self; we'll drop it
    k = min(topk + 1, n)

    nn = NearestNeighbors(
        n_neighbors=k, metric=metric, algorithm="brute", n_jobs=n_jobs
    )
    nn.fit(X)
    dists, idxs = nn.kneighbors(X, return_distance=True)
    # cosine distance in [0,2]; similarity = 1 - dist/2 (or 1 - dist if sklearn returns [0,1])
    if metric == "cosine":
        # sklearn cosine distance is in [0,2] when input not normalized; guard by normalizing explicitly:
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        Xn = X / norms
        nn.fit(Xn)
        dists, idxs = nn.kneighbors(Xn, return_distance=True)
        # sklearn cosine distance for normalized vectors is in [0,2], with 0==max sim
        # use sim = 1 - dist (since sklearn returns 1 - cosine_similarity for normalized vectors)
        sims = 1.0 - dists
    else:
        # Convert distances to an RBF-like similarity with auto bandwidth (median of non-zero)
        flat = dists.ravel()
        sigma = np.median(flat[flat > 0]) if np.any(flat > 0) else 1.0
        sims = np.exp(-(dists ** 2) / (sigma ** 2 + 1e-12))

    rows = []
    cols = []
    vals = []
    for i in range(n):
        for j, s in zip(idxs[i], sims[i]):
            if j == i:
                continue  # drop self
            rows.append(i)
            cols.append(j)
            vals.append(float(s))

    S = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    # keep exactly topk per row (robust if knn returns ties or small k)
    S = S.tolil()
    for i in range(n):
        row = S.rows[i]
        data = S.data[i]
        if not row:
            continue
        if len(row) > topk:
            # argpartition avoids full sort
            idx = np.argpartition(data, -topk)[-topk:]
            keep = set(row[j] for j in idx)
            S.rows[i] = [r for r in row if r in keep]
            S.data[i] = [d for r, d in zip(row, data) if r in keep]
    S = S.tocsr()
    S = _row_normalize_csr(S)
    return S

def _neighbors_from_csr(A: sp.csr_matrix):
    A = A.tocsr()
    n = A.shape[0]
    neigh = [[] for _ in range(n)]
    weights = [[] for _ in range(n)]
    for i in range(n):
        start, end = A.indptr[i], A.indptr[i + 1]
        neigh[i] = list(A.indices[start:end])
        weights[i] = list(A.data[start:end]) if A.data is not None and len(A.data) else [1.0] * (end - start)
    return neigh, weights

class _IFWalkIterable:
    """
    Streaming iterable that yields walks on-the-fly to Word2Vec.
    Avoids storing all walks in RAM.
    """
    def __init__(
        self,
        adj: sp.csr_matrix,
        S: sp.csr_matrix,
        num_walks: int,
        walk_length: int,
        lambda_sim: float,
        tau: float,
        seed: int,
    ):
        self.adj = adj.tocsr()
        self.S = S.tocsr()
        self.n = self.adj.shape[0]
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.lambda_sim = float(max(0.0, lambda_sim))
        self.tau = float(min(max(0.0, tau), 1.0))
        self.rng = np.random.RandomState(seed)
        self.N_adj, _ = _neighbors_from_csr(self.adj)
        self.N_sim, _ = _neighbors_from_csr(self.S)

    def __iter__(self):
        nodes = np.arange(self.n)
        for _ in range(self.num_walks):
            self.rng.shuffle(nodes)
            for start in nodes:
                cur = int(start)
                walk = [str(cur)]
                for _ in range(self.walk_length - 1):
                    # similarity teleport
                    if self.tau > 0 and self.rng.rand() < self.tau and len(self.N_sim[cur]) > 0:
                        cand = self.N_sim[cur]
                        # probs = normalized similarity row already (S row-normalized)
                        start_idx, end_idx = self.S.indptr[cur], self.S.indptr[cur + 1]
                        probs = self.S.data[start_idx:end_idx]
                        nxt = int(self.rng.choice(cand, p=probs / probs.sum()))
                    else:
                        cand = self.N_adj[cur]
                        if not cand:
                            break
                        # weights: 1 + lambda * S_norm_ij
                        start_idx, end_idx = self.S.indptr[cur], self.S.indptr[cur + 1]
                        sim_cols = self.S.indices[start_idx:end_idx]
                        sim_vals = self.S.data[start_idx:end_idx]
                        # map similarities for current neighbors
                        sim_map = {int(j): float(v) for j, v in zip(sim_cols, sim_vals)}
                        w = np.array([1.0 + self.lambda_sim * sim_map.get(int(j), 0.0) for j in cand], dtype=np.float32)
                        s = w.sum()
                        if s <= 0:
                            nxt = int(self.rng.choice(cand))
                        else:
                            nxt = int(self.rng.choice(cand, p=w / s))
                    walk.append(str(nxt))
                    cur = nxt
                yield walk

class IFWalk:
    """
    Individual-Fairness–biased random walks with streaming Word2Vec training.
    - Memory safe (no dense similarity, no giant walk list).
    """
    def __init__(self, representation_size=64, window_size=5, workers=4, seed=0):
        self.representation_size = int(representation_size)
        self.window_size = int(window_size)
        self.workers = int(workers)
        self.seed = int(seed)
        self.embeddings_ = None

    def fit(
        self,
        adj,              # scipy.sparse CSR adjacency (n x n) or convertible
        X,                # numpy features (n x d)
        walk_length=40,
        num_walks=10,
        topk=20,
        sim_metric="cosine",
        lambda_sim=1.0,
        tau=0.0,
        n_jobs=-1,
        w2v_epochs=5,
        w2v_negative=5,
        w2v_min_count=0,
    ):
        # Ensure CSR adjacency
        if not sp.issparse(adj):
            adj = sp.csr_matrix(adj)
        else:
            adj = adj.tocsr()

        # Build sparse kNN similarity safely
        S = build_similarity_knn(X, topk=topk, metric=sim_metric, n_jobs=n_jobs)

        # Streaming walk iterable
        corpus = _IFWalkIterable(
            adj=adj, S=S, num_walks=num_walks, walk_length=walk_length,
            lambda_sim=lambda_sim, tau=tau, seed=self.seed
        )

        # Two-pass training: build_vocab then train — still streamed
        model = Word2Vec(
            vector_size=self.representation_size,
            window=self.window_size,
            min_count=w2v_min_count,
            sg=1,
            negative=w2v_negative,
            workers=self.workers,
            seed=self.seed,
        )
        model.build_vocab(corpus_iterable=corpus)
        model.train(corpus_iterable=corpus, total_examples=model.corpus_count, epochs=w2v_epochs)

        # Export embeddings in node order
        n = adj.shape[0]
        emb = np.zeros((n, self.representation_size), dtype=np.float32)
        for i in range(n):
            emb[i] = model.wv[str(i)]
        self.embeddings_ = emb
        return self

    def get_embeddings(self):
        return self.embeddings_