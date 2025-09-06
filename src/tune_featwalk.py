# tune_featwalk.py

import torch
import numpy as np
import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from algorithms.FeatWalk import FeatWalk
from datasets import Nba

# -- Helper Functions (same as before) --
def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def calculate_individual_fairness(embeddings, features):
    feature_sim = cosine_similarity(features)
    np.fill_diagonal(feature_sim, 0)
    most_similar_indices = np.argmax(feature_sim, axis=1)
    embs1 = embeddings
    embs2 = embeddings[most_similar_indices]
    distances = np.sum((embs1 - embs2)**2, axis=1)
    if_score = np.mean(distances)
    return if_score

# -- Main Execution --

if __name__ == "__main__":
    setup_seed(42)

    # --- Hyperparameters to Tune ---
    ALPHA_VALUES_TO_TEST = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]
    
    print("Loading Nba dataset...")
    nba = Nba()
    adj, feats, idx_train, idx_val, idx_test, labels, sens = (
        nba.adj(), nba.features(), nba.idx_train(), nba.idx_val(),
        nba.idx_test(), nba.labels(), nba.sens(),
    )
    feats_np = feats.cpu().numpy()
    print("Dataset loaded successfully.\n")

    results = []

    # Loop through each alpha value and run FeatWalk
    for alpha_val in ALPHA_VALUES_TO_TEST:
        print(f"--- Running FeatWalk with alpha = {alpha_val} ---")
        model = FeatWalk()
        model.fit(adj_matrix=adj, feats=feats, labels=labels, idx_train=idx_train, sens=sens,
                  representation_size=64, number_walks=10, walk_length=20, alpha=alpha_val)
        
        (ACC, _, _, _, _, _, _, _, _, _, _) = model.predict(idx_test)
        if_score = calculate_individual_fairness(model.embs, feats_np)
        
        results.append({
            "Alpha": alpha_val,
            "Accuracy": ACC,
            "Ind. Fairness Score": if_score
        })
        print(f"--- Finished run for alpha = {alpha_val} ---\n")

    # --- Final Tuning Summary ---
    print("==========================================================")
    print("             FeatWalk Alpha Tuning Results")
    print("==========================================================")
    df = pd.DataFrame(results)
    df = df.set_index("Alpha")
    pd.options.display.float_format = '{:.4f}'.format
    print(df)
    print("\n* Look for the alpha that gives the lowest Ind. Fairness Score.")
    print("==========================================================")