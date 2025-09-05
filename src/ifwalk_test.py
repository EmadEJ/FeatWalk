import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from datasets import Bail
from debiasing.IFWalk import IFWalk
import scipy.sparse as sp
import torch

def to_csr(adj):
    if isinstance(adj, torch.Tensor):
        coo = adj.coalesce()
        i = coo.indices().cpu().numpy()
        v = coo.values().cpu().numpy()
        n = coo.size(0)
        return sp.csr_matrix((v, (i[0], i[1])), shape=(n, n))
    return adj.tocsr() if hasattr(adj, "tocsr") else sp.csr_matrix(adj)

def main():
    data = Bail()
    adj = to_csr(data.adj())
    X = data.features()
    y = data.labels()
    idx_train, idx_test = data.idx_train(), data.idx_test()

    # src/ifwalk_test.py (only the IFWalk.fit call changed)
    ifw = IFWalk(representation_size=64, window_size=5, workers=4, seed=42)
    ifw.fit(
        adj, X,
        walk_length=40,
        num_walks=6,  # ↓ slightly fewer walks to start
        topk=20,
        sim_metric='cosine',
        lambda_sim=1.0,
        tau=0.05,  # small teleport
        n_jobs=-1,  # use all cores for kNN search
        w2v_epochs=5,
        w2v_negative=5,
        w2v_min_count=0,
    )

    Z = ifw.get_embeddings()

    clf = LogisticRegression(max_iter=200)
    clf.fit(Z[idx_train], y[idx_train])
    y_pred = clf.predict(Z)
    acc = accuracy_score(y[idx_test], y_pred[idx_test])
    f1 = f1_score(y[idx_test], y_pred[idx_test], average='micro')
    print(f"Test Accuracy: {acc:.4f}, F1: {f1:.4f}")

    from metrics.metrics import IF, GDIF
    import torch

    # Convert to torch for IF function
    adj_torch = torch.tensor(adj.toarray()).to_sparse()
    y_hat = torch.tensor(y_pred, dtype=torch.float32).unsqueeze(1)  # shape [n,1]
    X_torch = torch.tensor(X, dtype=torch.float32)

    # Individual fairness
    if_val = IF(adj_torch, X_torch, y_hat)

    # Group disparity in IF
    gdif_val = GDIF(y_hat.detach().numpy(), data.sens())

    print(f"IF (lower=better): {if_val:.4f}")
    print(f"GDIF (closer to 1=better): {gdif_val:.4f}")


if __name__ == "__main__":
    main()
