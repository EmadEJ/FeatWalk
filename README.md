# FeatWalk: A Feature-Aware Approach to Individual Fairness in Graph Embeddings


This repository contains the official implementation for the **FeatWalk** algorithm, a novel random walk-based method designed to enhance **individual fairness** in node embeddings. It also includes implementations of baseline models like `CrossWalk` and `FairWalk` for comparison.

The core idea of `FeatWalk` is to guide the random walk process not just by the graph's structure, but also by the feature similarity of the nodes.

**Standard Walk:** `Node` ➔ `Follows Structure` ➔ `Next Node`  
**FeatWalk:** `Node` 🧠 ➔ `Checks Feature Similarity of Neighbors` ➔ `Follows Structure + Similarity` ➔ `Next Node`

---
## 🚀 Run on Google Colab

To get started immediately without any local setup, you can run the experiments directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18NuyDVUwC_I9YIQwyEkDyPJRXVs9e5mr#scrollTo=GtGLcdH0v33u)

---

## 🔧 Quick Start (Local Setup)

Follow these steps to set up the project on your local machine.

#### 1. Clone the Repository
```bash
git clone [https://github.com/EmadEJ/fair_graph_embedding.git](https://github.com/EmadEJ/fair_graph_embedding.git)
cd fair_graph_embedding
```

#### 2. Create and Activate a Virtual Environment
This project requires a clean, isolated environment.

```bash
# Create the environment
python3 -m venv .venv

# Activate the environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

#### 3. Install Dependencies
With the environment active, install the project in editable mode along with its dependencies. This command includes the necessary links to find compatible versions of PyTorch and its related libraries.

```bash
# Upgrade pip and install pandas for results tables
pip install --upgrade pip
pip install pandas

# Install the project and its core dependencies
pip install -e . -f [https://data.pyg.org/whl/torch-1.12.0%2Bcu116.html](https://data.pyg.org/whl/torch-1.12.0%2Bcu116.html) -f [https://download.pytorch.org/whl/torch_stable.html](https://download.pytorch.org/whl/torch_stable.html)  -f [https://data.dgl.ai/wheels/cu116/repo.html](https://data.dgl.ai/wheels/cu116/repo.html) -f [https://data.dgl.ai/wheels-test/repo.html](https://data.dgl.ai/wheels-test/repo.html)
```

---

## 📊 Running the Comparison Experiments

You can run the full comparison between `CrossWalk`, `FairWalk`, and `FeatWalk` on different datasets using the provided scripts.

```bash
# Example: Run the stable comparison on the Facebook dataset
python compare_facebook.py

# Example: Run the stable comparison on the German dataset
python compare_german.py
```

---

## 🙏 Acknowledgments

This codebase is heavily derived from the excellent **`PyGDebias`** library. Our work builds upon their data loaders and baseline implementations. We highly recommend checking out the original repository for a broader collection of fairness algorithms in graph machine learning.

* **Original `PyGDebias` Repository:** [https://github.com/yushundong/PyGDebias](https://github.com/yushundong/PyGDebias)