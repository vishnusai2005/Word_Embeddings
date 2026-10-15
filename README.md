# Word2Vec from Scratch in PyTorch

**CBOW and Skip-gram with Negative Sampling, implemented without gensim or any pre-built embedding library.**

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![NLP](https://img.shields.io/badge/Domain-NLP-green)
![Status](https://img.shields.io/badge/Status-Educational%20Prototype-lightgrey)

---

## Overview

Word embeddings are the foundation of modern NLP, from search and recommendation to the token embeddings inside large language models. This project rebuilds the two classic Word2Vec architectures **from first principles**, so that every step is explicit and inspectable: tokenization, vocabulary construction, training-pair generation, noise distribution, model definition, loss computation, and the training loop.

The goal is not to beat a pre-trained model. The goal is to demonstrate a working understanding of how embeddings are actually learned.

## What This Project Demonstrates

| Skill | Evidence in the notebook |
|---|---|
| NLP preprocessing | Custom regex tokenizer, sentence splitter, frequency-based vocabulary (`stoi` / `itos`) |
| Training-data engineering | Sliding-window generation of CBOW `(context -> target)` and Skip-gram `(center -> context)` pairs |
| Probabilistic sampling | Negative sampling from the unigram distribution raised to the 3/4 power, as in Mikolov et al. (2013) |
| Deep learning in PyTorch | Custom `nn.Module` classes, masked mean pooling over variable-length contexts, `einsum` for batched scoring |
| Optimization | Binary cross-entropy with logits, Adam optimizer, small-uniform embedding initialization for stable training |
| Reproducibility | Fixed seeds for `random`, `numpy`, and `torch` |

## Models Implemented

### CBOW with Negative Sampling (`CBOWNeg`)
Averages the embeddings of the surrounding context words (with a padding mask for variable window sizes) and scores the result against the true target word and `k` sampled negative words.

### Skip-gram with Negative Sampling (`SGNS`)
Takes a center word and scores it against each true context word and `k` sampled negatives, using separate input (center) and output (context) embedding tables.

## Pipeline

```
Raw text -> tokenize -> vocabulary (51 words) -> window pairs
         -> negative sampling (unigram^0.75) -> PyTorch model -> BCE loss -> Adam
```

## Configuration

| Setting | Value |
|---|---|
| Corpus | 7-line hand-written demo corpus (royalty, food, animals, NLP) |
| Vocabulary size | 51 |
| Window size | 2 |
| Training pairs | 81 (CBOW), 234 (Skip-gram) |
| Embedding dimension | 64 |
| Negative samples per positive (`k`) | 5 |
| Optimizer / learning rate | Adam / 3e-3 |
| Batch size | 64 |
| Epochs | 8 (CBOW), 6 (Skip-gram) |
| Random seed | 42 |

## Results

| Model | Epoch 1 loss | Final loss |
|---|---|---|
| CBOW + Negative Sampling | 1.3861 | 1.3250 (epoch 8) |
| Skip-gram + Negative Sampling | 1.3863 | 1.3638 (epoch 6) |

Both models train in the expected direction, with loss decreasing every epoch. Note that the starting value of about 1.386 equals `2 * ln(2)`, which is exactly the loss of a model that outputs 0.5 for every pair. See the audit below for what this implies.

## Honest Audit: Scope, Known Issues, and Next Steps

This section is deliberate. A portfolio project is more useful when its limits are stated clearly.

**Scope**
- This is an **educational implementation on a toy corpus**, not a trained, production-grade embedding model. The corpus is only a handful of sentences, so the learned vectors are not expected to show reliable semantic structure (for example, king - man + woman ≈ queen).
- Training is short and stops close to the chance-level loss, so the models are **under-trained**. No embedding-quality evaluation (nearest neighbors, cosine similarity, analogy tests, PCA/t-SNE plot) has been run yet.

**Known issues found during review**
1. **CBOW embedding-table inconsistency.** In `CBOWNeg.forward`, the positive target is looked up in `in_emb`, while negatives use `out_emb`. In standard Word2Vec the target and negatives should both come from the output table (as `SGNS` already does). Fix: use `self.out_emb(targets)`.
2. **Duplicate line.** `self.out_emb` is defined twice in `CBOWNeg.__init__`. Harmless, but redundant.
3. **Typo in helper.** `pca_2d` uses `keedpdims` instead of `keepdims`, so it would raise a `TypeError` when called. It has not been executed in the notebook yet.
4. **Negative sampler can return the true word.** On a 51-word vocabulary, sampled negatives can coincide with the real target or context, adding label noise. Fix: resample or mask collisions.
5. **Naming and cleanliness.** The variable `poss_loss` should be `pos_loss`; several imports (`math`, `defaultdict`, `Dict`) and plotting helpers (`plot_bar`, `plot_heatmap`) are currently unused.

**Roadmap**
- [ ] Apply the fixes above
- [ ] Train on a real corpus (for example, a movie-review dataset such as IMDB) with a minimum-frequency cutoff and subsampling of frequent words
- [ ] Add evaluation: top-k nearest neighbors, cosine-similarity heatmap, analogy tests, 2D PCA visualization
- [ ] Compare against `gensim.models.Word2Vec` as a reference baseline
- [ ] Use the learned embeddings as features for a downstream task such as sentiment classification

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Install dependencies
pip install torch numpy matplotlib jupyter

# 3. Launch the notebook
jupyter notebook word2vec.ipynb
```

## Tech Stack

Python · PyTorch · NumPy · Matplotlib · Jupyter

## Reference

Mikolov, T., Sutskever, I., Chen, K., Corrado, G., and Dean, J. (2013). *Distributed Representations of Words and Phrases and their Compositionality.* NeurIPS.

## Author

**Vishnusai Vydhyam**
Final-year B.Tech, Computer Science Engineering (AI & ML), Mohan Babu University, Tirupati
Seeking ML, Data Science, and AI Engineer roles.

<!-- Add your LinkedIn / email / portfolio links here -->
