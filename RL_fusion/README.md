# RL-based Dynamic Fusion

Reinforcement-learning variant for fusing multi-view media representations, from
**"A Multi-View Media Profiling Suite: Resources, Evaluation, and Analysis"**
(Findings of the ACL 2026).

Instead of combining the five views with a fixed strategy (concatenation,
averaging, attention, ...), an RL agent learns **outlet-specific weights** for
each view and produces a single fused embedding used for bias / factuality
prediction.

## Formulation (contextual bandit)

For a media outlet `F_i`, the state is the concatenation of its five view
embeddings:

```
s_t = { F^(a), F^(h), F^(l), F^(t), F^(w) }
```

| Symbol | View | Source |
|--------|------|--------|
| `F^(a)` | Alexa graph      | GNN over audience-overlap graph |
| `F^(h)` | Hyperlink graph  | GNN over on-site hyperlink graph |
| `F^(l)` | LLM graph        | GNN over LLM-generated graph |
| `F^(t)` | Articles         | text embedding of outlet articles |
| `F^(w)` | Wikipedia        | text embedding of Wikipedia description |

The action is a continuous weight vector `w ∈ R^5`, `w_k ∈ [0, 1]`. The fused
embedding is `E_fused = Σ_k w_k F^(k)`, and the reward is the probability of the
true label under a **fixed** classifier, `r_t = P(y_true | E_fused)`. Because
actions do not affect state transitions, the problem is a contextual bandit and
the discount factor is set to **0** (immediate reward only). The policy is
trained with **PPO** (Stable-Baselines3) in a **Gymnasium** environment.

## Files

| File | Purpose |
|------|---------|
| `dynamic_fusion_env.py` | Gymnasium environment (`DynamicFusionEnv`) |
| `main.py` | PPO/SAC/TRPO training + evaluation script |
| `requirements.txt` | Python dependencies |
| `data/` | Labels/splits (included) + embedding data (download separately) |

## Setup

```bash
pip install -r requirements.txt
```

## Data

The label and split files are included under `data/`:

- `data/corpus.tsv` — outlet labels (bias / factuality)
- `data/splits.json` — 5-fold cross-validation splits

The **embedding data is large (~2.3 GB)** and is **not** stored in this repo.
Download it and place the files/folders directly inside `data/` (or point to
another location with the `DYNAMIC_FUSION_DATA_DIR` environment variable):

> **Download the embedding data (OneDrive, ~2.2 GB):**
> [browse and download the files here](https://utsacloud-my.sharepoint.com/:f:/g/personal/muhammadumer_siddique_my_utsa_edu/IgDIJX8--wYeQ5r7vjseJVc-AWB68mLAx1uLj78_mWjs3o8?e=L9cIbZ)

## Usage

Train and evaluate a PPO agent (ACL-2020, bias task):

```bash
python main.py --algo ppo --datasetsize small --task bias
```

Other options:

```bash
python main.py --algo ppo --datasetsize large --task fact --split-id 0
```
