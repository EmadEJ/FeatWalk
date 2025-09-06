# compare_nba_stable.py

import torch
import numpy as np
import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Import all three of your algorithms and the dataset
from algorithms.CrossWalk import CrossWalk
from algorithms.FairWalk import FairWalk
from algorithms.FeatWalk import FeatWalk
from datasets import Nba

# -- Helper Functions --

def setup_seed(seed):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def calculate_individual_fairness(embeddings, features):
    """Calculates the IF score for any set of embeddings. Lower is better."""
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
    
    # --- Hyperparameters for Consistency ---
    NUM_WALKS = 40
    WALK_LENGTH = 20
    DIMS = 64
    ALPHA = 1.0 # For FeatWalk

    # 1. Load the dataset once
    print("Loading Nba dataset...")
    nba = Nba()
    adj, feats, idx_train, idx_val, idx_test, labels, sens = (
        nba.adj(),
        nba.features(),
        nba.idx_train(),
        nba.idx_val(),
        nba.idx_test(),
        nba.labels(),
        nba.sens(),
    )
    feats_np = feats.cpu().numpy()
    print("Dataset loaded successfully.\n")

    results = []

    # --- Run 1: CrossWalk ---
    print(f"--- Running CrossWalk (num_walks={NUM_WALKS}) ---")
    crosswalk_model = CrossWalk()
    crosswalk_model.fit(adj_matrix=adj, feats=feats, labels=labels, idx_train=idx_train, sens=sens,
                        representation_size=DIMS, number_walks=NUM_WALKS, walk_length=WALK_LENGTH)
    (ACC, AUCROC, F1, _, _, _, _, _, _, SP, EO) = crosswalk_model.predict(idx_test, idx_val)
    if_score = calculate_individual_fairness(crosswalk_model.embs, feats_np)
    results.append({
        "Algorithm": "CrossWalk", "Accuracy": ACC, "Ind. Fairness Score": if_score,
        "Stat. Parity (SP)": SP, "Equal Opp. (EO)": EO
    })
    print("--- CrossWalk Complete ---\n")

    # --- Run 2: FairWalk ---
    print(f"--- Running FairWalk (num_walks={NUM_WALKS}) ---")
    fairwalk_model = FairWalk()
    fairwalk_model.fit(adj=adj, labels=labels, idx_train=idx_train, sens=sens,
                       dimensions=DIMS, num_walks=NUM_WALKS, walk_length=WALK_LENGTH, workers=4)
    (ACC, AUCROC, F1, _, _, _, _, _, _, SP, EO) = fairwalk_model.predict(idx_test, idx_val)
    if_score = calculate_individual_fairness(fairwalk_model.embs, feats_np)
    results.append({
        "Algorithm": "FairWalk", "Accuracy": ACC, "Ind. Fairness Score": if_score,
        "Stat. Parity (SP)": SP, "Equal Opp. (EO)": EO
    })
    print("--- FairWalk Complete ---\n")

    # --- Run 3: FeatWalk ---
    print(f"--- Running FeatWalk (num_walks={NUM_WALKS}) ---")
    featwalk_model = FeatWalk()
    featwalk_model.fit(adj_matrix=adj, feats=feats, labels=labels, idx_train=idx_train, sens=sens,
                       representation_size=DIMS, number_walks=NUM_WALKS, walk_length=WALK_LENGTH, alpha=ALPHA)
    (ACC, AUCROC, F1, _, _, _, _, _, _, SP, EO) = featwalk_model.predict(idx_test)
    if_score = calculate_individual_fairness(featwalk_model.embs, feats_np)
    results.append({
        "Algorithm": "FeatWalk", "Accuracy": ACC, "Ind. Fairness Score": if_score,
        "Stat. Parity (SP)": SP, "Equal Opp. (EO)": EO
    })
    print("--- FeatWalk Complete ---\n")

    # --- 4. Final Comparison Summary ---
    print("==========================================================")
    print("                 Algorithm Comparison")
    print("==========================================================")
    df = pd.DataFrame(results)
    df = df.set_index("Algorithm")
    pd.options.display.float_format = '{:.4f}'.format
    print(df)
    print("\n* Individual Fairness Score: Lower is better.")
    print("* SP and EO: Values closer to 0.0 are better.")
    print("==========================================================")