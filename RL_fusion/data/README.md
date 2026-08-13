# Data directory

This folder holds the inputs for the RL-based Dynamic Fusion environment.

## Included (ACL-2020 metadata)

- `corpus.tsv` — outlet labels (bias / factuality)
- `splits.json` — 5-fold cross-validation splits

## Download separately (large embedding data, ~2.3 GB)

The precomputed embeddings are too large to store in the repository. Download
them and place the files/folders directly in this directory.

> **Download the embedding data (OneDrive, ~2.2 GB):**
> [browse and download the files here](https://utsacloud-my.sharepoint.com/:f:/g/personal/muhammadumer_siddique_my_utsa_edu/IgDIJX8--wYeQ5r7vjseJVc-AWB68mLAx1uLj78_mWjs3o8?e=L9cIbZ)

Required files/folders:

| Item | View / role |
|------|-------------|
| `embeddings_1a_small.pkl` | Articles (ACL-2020) |
| `embeddings_1b_small.pkl` | Wikipedia (ACL-2020) |
| `embeddings_2a_large.pkl` | Articles (MBFC-2025) |
| `embeddings_2b_large.pkl` | Wikipedia (MBFC-2025) |
| `Alexa/` | Alexa-graph GNN embeddings |
| `gpt_small/`, `gpt_large/` | LLM-graph GNN embeddings |
| `onsite_small/`, `onsite_large/` | Hyperlink-graph GNN embeddings |
| `mbfc2025_large_labels.pkl` | MBFC-2025 bias and factuality labels |
| `Splits/Large/` | Official MBFC-2025 train/validation/test splits |

You can also keep the data elsewhere and set `DYNAMIC_FUSION_DATA_DIR` to that
path instead of copying files here.
