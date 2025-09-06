# fairwalk_example.py (Corrected)

import torch
import numpy as np
import random

# 1. Import the FairWalk model and a small dataset
from algorithms import FairWalk
from datasets import Nba

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
# We use the small Nba dataset for a quick demonstration.
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


# 4. Initialize and train the FairWalk model
model = FairWalk()

print("\nTraining FairWalk model...")
# Note the corrected parameters in the .fit() call below
model.fit(adj=adj,
          labels=labels,
          idx_train=idx_train,
          sens=sens,
          walk_length=20,
          num_walks=10,         # Corrected parameter name
          dimensions=128,
          epochs=5)       # Corrected parameter name
print("Training complete.")


# 5. Evaluate the model on the test set
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
print(f"AUC-ROC (Sensitive Group 0): {AUCROC_sens0 if isinstance(AUCROC, str) else AUCROC_sens0:.4f}")
print(f"F1 Score (Sensitive Group 0): {F1_sens0:.4f}")
print("--------------------------")
print(f"Accuracy (Sensitive Group 1): {ACC_sens1:.4f}")
print(f"AUC-ROC (Sensitive Group 1): {AUCROC_sens1 if isinstance(AUCROC_sens1, str) else AUCROC_sens1:.4f}")
print(f"F1 Score (Sensitive Group 1): {F1_sens1:.4f}")
print("--------------------------")
print(f"Statistical Parity (SP): {SP:.4f}")
print(f"Equal Opportunity (EO): {EO:.4f}")
print("--- End of Report ---")