# src/ifwalk_test.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from datasets import Bail
from debiasing.IFWalk import IFWalk

import scipy.sparse as sp
import torch

# ---- CPU-safe IF (imports only) ----
from metrics.metrics import IF_cpu  # <-- make sure IF_cpu is added as shown below


def to_csr(adj):
    # Convert torch sparse -> scipy CSR if needed
    if isinstance(adj, torch.Tensor) and adj.is_sparse:
        coo = adj.coalesce()
        i = coo.indices().cpu().numpy()
        v = coo.values().cpu().numpy()
        n = coo.size(0)
        return sp.csr_matrix((v, (i[0], i[1])), shape=(n, n))
    # If already scipy sparse
    if hasattr(adj, "tocsr"):
        return adj.tocsr()
    return sp.csr_matrix(adj)


def main():
    # 1) Load dataset
    data = Bail()
    adj = to_csr(data.adj())
    X = data.features()
    y = data.labels()
    sens = data.sens()
    idx_train, idx_test = data.idx_train(), data.idx_test()

    # 2) Train IFWalk embeddings (tweak knobs to your device)
    ifw = IFWalk(representation_size=64, window_size=5, workers=4, seed=42)
    ifw.fit(
        adj,
        X,
        walk_length=40,
        num_walks=6,
        topk=15,
        sim_metric="cosine",
        lambda_sim=1.0,
        tau=0.05,
        n_jobs=-1,
        w2v_epochs=5,
        w2v_negative=5,
        w2v_min_count=0,
    )
    Z = ifw.get_embeddings()

    # 3) Downstream classifier (stable vs LP)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Z[idx_train], y[idx_train])
    y_pred = clf.predict(Z)
    y_prob = None
    try:
        y_prob = clf.predict_proba(Z)[:, 1]
    except Exception:
        pass

    # 4) Task metrics
    acc = accuracy_score(y[idx_test], y_pred[idx_test])
    f1 = f1_score(y[idx_test], y_pred[idx_test], average="micro")
    print(f"Test Accuracy: {acc:.4f}, F1: {f1:.4f}")

    # 5) Fairness: IF on CPU (pass probabilities if available)
    #    IF uses smoothness of predictions across a similarity graph.
    y_hat_for_if = y_prob if y_prob is not None else y_pred.astype(float)
    if_val = IF_cpu(adj, X, y_hat_for_if, topk=20, metric="cosine")
    print(f"IF (lower=better): {if_val:.6f}")


if __name__ == "__main__":
    main()
