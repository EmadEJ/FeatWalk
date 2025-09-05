# compare_algorithms_facebook.py

import torch
import numpy as np
import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Import all three of your algorithms and the Facebook dataset
from debiasing.CrossWalk import CrossWalk
from debiasing.FairWalk import FairWalk
from debiasing.FeatWalk import FeatWalk
from datasets import Facebook

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

    # 1. Load the Facebook dataset
    print("Loading Facebook dataset...")
    facebook = Facebook()
    adj, feats, idx_train, idx_val, idx_test, labels, sens = (
        facebook.adj(),
        facebook.features(),
        facebook.idx_train(),
        facebook.idx_val(),
        facebook.idx_test(),
        facebook.labels(),
        facebook.sens(),
    )
    feats_np = feats.cpu().numpy()
    print("Dataset loaded successfully.\n")

    results = []

    # --- Run 1: CrossWalk (Baseline) ---
    print("--- Running CrossWalk (Baseline) ---")
    crosswalk_model = CrossWalk()
    crosswalk_model.fit(adj_matrix=adj, feats=feats, labels=labels, idx_train=idx_train, sens=sens,
                        representation_size=64, number_walks=10, walk_length=20)
    (ACC, AUCROC, F1, _, _, _, _, _, _, SP, EO) = crosswalk_model.predict(idx_test, idx_val)
    if_score_cw = calculate_individual_fairness(crosswalk_model.embs, feats_np)
    results.append({
        "Algorithm": "CrossWalk", "Accuracy": ACC, "Ind. Fairness Score": if_score_cw,
        "Stat. Parity (SP)": SP, "Equal Opp. (EO)": EO
    })
    print("--- CrossWalk Complete ---\n")

    # --- Run 2: FairWalk (Group Fairness Baseline) ---
    print("--- Running FairWalk (Group Fairness) ---")
    fairwalk_model = FairWalk()
    fairwalk_model.fit(adj=adj, labels=labels, idx_train=idx_train, sens=sens,
                       dimensions=64, num_walks=10, walk_length=20, workers=4)
    (ACC, AUCROC, F1, _, _, _, _, _, _, SP, EO) = fairwalk_model.predict(idx_test, idx_val)
    if_score_fw = calculate_individual_fairness(fairwalk_model.embs, feats_np)
    results.append({
        "Algorithm": "FairWalk", "Accuracy": ACC, "Ind. Fairness Score": if_score_fw,
        "Stat. Parity (SP)": SP, "Equal Opp. (EO)": EO
    })
    print("--- FairWalk Complete ---\n")

    # --- Run 3: FeatWalk (Individual Fairness) ---
    print("--- Running FeatWalk (Individual Fairness) ---")
    featwalk_model = FeatWalk()
    featwalk_model.fit(adj_matrix=adj, feats=feats, labels=labels, idx_train=idx_train, sens=sens,
                       representation_size=64, number_walks=10, walk_length=20, alpha=5.0)
    (ACC, AUCROC, F1, _, _, _, _, _, _, SP, EO) = featwalk_model.predict(idx_test)
    if_score_ftw = calculate_individual_fairness(featwalk_model.embs, feats_np)
    results.append({
        "Algorithm": "FeatWalk", "Accuracy": ACC, "Ind. Fairness Score": if_score_ftw,
        "Stat. Parity (SP)": SP, "Equal Opp. (EO)": EO
    })
    print("--- FeatWalk Complete ---\n")

    # --- 4. Final Comparison Summary ---
    print("==========================================================")
    print("       Algorithm Comparison on Facebook Dataset")
    print("==========================================================")
    
    # You may need to run 'pip install pandas' for this table.
    df = pd.DataFrame(results)
    df = df.set_index("Algorithm")
    
    pd.options.display.float_format = '{:.4f}'.format
    
    print(df)
    print("\n* Individual Fairness Score: Lower is better.")
    print("* SP and EO: Values closer to 0.0 are better.")
    print("==========================================================")