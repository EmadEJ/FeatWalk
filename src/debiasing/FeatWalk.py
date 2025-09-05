# FeatWalk.py

import torch
import numpy as np
import random
from tqdm import tqdm
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

class FeatWalk:
    def _generate_walks(self, adj_dict, feature_sim, alpha):
        """
        Generates feature-guided random walks.
        alpha: Controls the strength of the feature similarity bias.
        """
        walks = []
        nodes = list(adj_dict.keys())
        # Use tqdm for a progress bar over the number of walks
        for _ in tqdm(range(self.number_walks), desc="Generating Walks"):
            random.shuffle(nodes)
            for node in nodes:
                walk = [node]
                while len(walk) < self.walk_length:
                    current_node = walk[-1]
                    neighbors = adj_dict[current_node]
                    if len(neighbors) == 0:
                        break
                    
                    # Calculate transition weights based on feature similarity
                    weights = []
                    for neighbor in neighbors:
                        # Baseline weight is 1, biased by feature similarity
                        similarity = feature_sim[current_node, neighbor]
                        weight = 1.0 + alpha * similarity
                        weights.append(weight)
                    
                    # Normalize weights to get probabilities
                    probabilities = np.array(weights) / np.sum(weights)
                    
                    # Choose the next node based on the biased probabilities
                    next_node = np.random.choice(neighbors, p=probabilities)
                    walk.append(next_node)
                walks.append([str(n) for n in walk])
        return walks

    def fit(self, adj_matrix, feats, labels, idx_train, sens,
            walk_length=20, number_walks=10, representation_size=64,
            alpha=5.0):
        """
        Trains the FeatWalk model.
        alpha: A new hyperparameter to control the feature-guidance strength.
        """
        self.walk_length = walk_length
        self.number_walks = number_walks
        
        print("Pre-calculating feature similarity matrix...")
        feature_sim = cosine_similarity(feats.cpu().numpy())

        print("Creating adjacency dictionary...")
        adj_matrix = adj_matrix.coalesce() 
        indices = adj_matrix.indices()
        adj_dict = {i: [] for i in range(adj_matrix.shape[0])}
        for i, j in zip(indices[0].tolist(), indices[1].tolist()):
            adj_dict[i].append(j)

        # 1. Generate walks using the new feature-guided strategy
        walks = self._generate_walks(adj_dict, feature_sim, alpha)
        
        # 2. Train a Word2Vec model on the generated walks
        print("Training Word2Vec model on walks...")
        model = Word2Vec(walks, vector_size=representation_size, window=5,
                         min_count=0, sg=1, workers=4)
        
        # Store embeddings in a sorted array
        self.embs = np.zeros((adj_matrix.shape[0], representation_size))
        for i in range(adj_matrix.shape[0]):
            if str(i) in model.wv:
                self.embs[i] = model.wv[str(i)]
        
        self.labels = labels
        self.sens = sens.squeeze()
        
        # 3. Train a downstream classifier
        print("Training downstream classifier...")
        self.lgreg = LogisticRegression(
            random_state=0, C=1.0, multi_class="auto", solver="lbfgs", max_iter=1000
        ).fit(self.embs[idx_train], labels[idx_train])

    # The predict method and its helpers remain the same as in IndFairWalk
    def predict(self, idx_test):
        pred = self.lgreg.predict(self.embs[idx_test])
        y_test = self.labels[idx_test]
        z_test = self.sens[idx_test]

        F1 = f1_score(y_test, pred, average="micro")
        ACC = accuracy_score(y_test, pred)
        try: AUCROC = roc_auc_score(y_test, pred)
        except: AUCROC = "N/A"
        
        ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1 = self.predict_sens_group(pred, y_test, z_test)
        SP, EO = self.fair_metric(pred, y_test, z_test)
        
        return ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, SP, EO

    def fair_metric(self, pred, labels, sens):
        idx_s0 = sens == 0
        idx_s1 = sens == 1
        idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
        idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)
        parity = abs(sum(pred[idx_s0]) / sum(idx_s0) - sum(pred[idx_s1]) / sum(idx_s1))
        equality = abs(sum(pred[idx_s0_y1]) / sum(idx_s0_y1) - sum(pred[idx_s1_y1]) / sum(idx_s1_y1))
        return parity.item(), equality.item()

    def predict_sens_group(self, y_pred, y_test, z_test):
        result = []
        for sens_val in [0, 1]:
            # Use .sum() on torch tensors
            if (z_test == sens_val).sum() == 0:
                result.extend([0, "N/A", 0])
                continue
            F1 = f1_score(y_test[z_test == sens_val], y_pred[z_test == sens_val], average="micro")
            ACC = accuracy_score(y_test[z_test == sens_val], y_pred[z_test == sens_val])
            try: AUCROC = roc_auc_score(y_test[z_test == sens_val], y_pred[z_test == sens_val])
            except: AUCROC = "N/A"
            result.extend([ACC, AUCROC, F1])
        return result