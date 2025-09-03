# crosswalk_example.py

import torch
import numpy as np
import random

# 1. Import the CrossWalk model and a dataset
# Assuming CrossWalk.py is in the same directory or accessible in the python path.
from debiasing import CrossWalk
from datasets import Bail # We'll use the same dataset for a fair comparison.


# 2. Function to ensure reproducibility
def setup_seed(seed):
    """
    Sets the random seed for reproducibility across different libraries.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

# Set a seed for consistent results
setup_seed(42)


# 3. Load the dataset
# The Bail dataset is a standard benchmark for fairness in graph ML.
print("Loading Bail dataset...")
bail = Bail()
adj, feats, idx_train, idx_val, idx_test, labels, sens = (
    bail.adj(),
    bail.features(),
    bail.idx_train(),
    bail.idx_val(),
    bail.idx_test(),
    bail.labels(),
    bail.sens(),
)
print("Dataset loaded successfully.")


# 4. Initialize and train the CrossWalk model
# CrossWalk is not an end-to-end GNN. It first learns node embeddings
# using biased random walks and then trains a classifier on them.
model = CrossWalk()

# The `fit` method will perform the random walks to generate embeddings
# and then train a logistic regression classifier on the training nodes.
print("\nTraining CrossWalk model...")
# These parameters can be tuned for better performance.
model.fit(adj_matrix=adj,
          feats=feats,
          labels=labels,
          idx_train=idx_train,
          sens=sens,
          walk_length=20,       # Length of each random walk
          number_walks=10,      # Number of walks per node
          representation_size=128,
          epochs=5) # Dimensionality of the node embeddings
print("Training complete.")


# 5. Evaluate the model on the test set
# The `predict` method uses the learned embeddings to classify the test nodes
# and calculates various performance and fairness metrics.
print("\nEvaluating model on the test set...")
(
    ACC,
    AUCROC,
    F1,
    ACC_sens0,
    AUCROC_sens0,
    F1_sens0,
    ACC_sens1,
    AUCROC_sens1,
    F1_sens1,
    SP,
    EO,
) = model.predict(idx_test, idx_val)


# 6. Print the results
print("\n--- Evaluation Results ---")
print(f"Overall Accuracy (ACC): {ACC:.4f}")
print(f"Overall AUC-ROC: {AUCROC if isinstance(AUCROC, str) else AUCROC:.4f}")
print(f"Overall F1 Score: {F1:.4f}")
print("--------------------------")
print(f"Accuracy (Sensitive Group 0): {ACC_sens0:.4f}")
print(f"AUC-ROC (Sensitive Group 0): {AUCROC_sens0 if isinstance(AUCROC_sens0, str) else AUCROC_sens0:.4f}")
print(f"F1 Score (Sensitive Group 0): {F1_sens0:.4f}")
print("--------------------------")
print(f"Accuracy (Sensitive Group 1): {ACC_sens1:.4f}")
print(f"AUC-ROC (Sensitive Group 1): {AUCROC_sens1 if isinstance(AUCROC_sens1, str) else AUCROC_sens1:.4f}")
print(f"F1 Score (Sensitive Group 1): {F1_sens1:.4f}")
print("--------------------------")
print(f"Statistical Parity (SP): {SP:.4f}")
print(f"Equal Opportunity (EO): {EO:.4f}")
print("--- End of Report ---")