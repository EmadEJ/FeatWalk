# crosswalk_baseline_example.py

import torch
import numpy as np
import random
from debiasing.CrossWalk import CrossWalk # <-- Using the original CrossWalk
from datasets import Nba
from sklearn.metrics.pairwise import cosine_similarity

# 2. Function to ensure reproducibility
def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def calculate_individual_fairness(embeddings, features):
    """
    Helper function to calculate the IF score for any set of embeddings.
    Lower is better.
    """
    print("Calculating Individual Fairness Score...")
    # Calculate pairwise similarity based on node features
    feature_sim = cosine_similarity(features)
    np.fill_diagonal(feature_sim, 0) # Ignore self-similarity

    # Find the most similar neighbor for each node based on features
    most_similar_indices = np.argmax(feature_sim, axis=1)
    
    # Get embeddings for the original nodes and their most similar counterparts
    embs1 = embeddings
    embs2 = embeddings[most_similar_indices]
    
    # Calculate the squared Euclidean distance and average it
    distances = np.sum((embs1 - embs2)**2, axis=1)
    if_score = np.mean(distances)
    return if_score


setup_seed(42)

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
print("Dataset loaded successfully.")

# We need the original features in numpy format for the IF calculation
feats_np = feats.cpu().numpy()

model = CrossWalk()

print("\nTraining original CrossWalk model...")
# Note: The original fit method in CrossWalk.py has a hardcoded 50 epochs
model.fit(adj_matrix=adj,
          feats=feats,
          labels=labels,
          idx_train=idx_train,
          sens=sens,
          walk_length=20,
          number_walks=10,
          representation_size=64)
print("Training complete.")

print("\nEvaluating model on the test set...")
(
    ACC, AUCROC, F1,
    ACC_sens0, AUCROC_sens0, F1_sens0,
    ACC_sens1, AUCROC_sens1, F1_sens1,
    SP, EO,
) = model.predict(idx_test, idx_val)

# Calculate the Individual Fairness metric on the embeddings from CrossWalk
if_score = calculate_individual_fairness(model.embs, feats_np)

print("\n--- BASELINE Evaluation Results (CrossWalk) ---")
print(f"Overall Accuracy (ACC): {ACC:.4f}")
print(f"Overall AUC-ROC: {AUCROC if isinstance(AUCROC, str) else AUCROC:.4f}")
print(f"Overall F1 Score: {F1:.4f}")
print("--------------------------")
print(f"Statistical Parity (SP): {SP:.4f}")
print(f"Equal Opportunity (EO): {EO:.4f}")
print("--- INDIVIDUAL FAIRNESS ---")
print(f"Individual Fairness Score (lower is better): {if_score:.4f}")
print("--- End of Report ---")