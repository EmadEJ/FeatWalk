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
git clone https://github.com/EmadEJ/FeatWalk.git
cd FeatWalk
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

```bash
# Install the project and its core dependencies
pip install -r requirements.txt
```

---

## 📊 Running the Comparison Experiments

You can run the full comparison between `CrossWalk`, `FairWalk`, and `FeatWalk` on different datasets using the provided scripts.

```bash
# Enter the source code directory
cd src

# Example: Run the stable comparison on the Facebook dataset
python compare_facebook.py

# Example: Run the stable comparison on the German dataset
python compare_german.py
```

---

## 🙏 Acknowledgments

This codebase is heavily derived from the excellent **`PyGDebias`** library. Our work builds upon their data loaders and baseline implementations. We highly recommend checking out the original repository for a broader collection of fairness algorithms in graph machine learning.

* **Original `PyGDebias` Repository:** [https://github.com/yushundong/PyGDebias](https://github.com/yushundong/PyGDebias)