# Results

This directory contains the output of the retrieval and aggregation pipeline stages. Files are excluded from version control by default (see `.gitignore`).

```
results/
└── dataset_{1,2,3}/
    ├── results_llm={llm}_embedding={emb}_scorer={scorer}_m={m}.parquet   # retrieval pairs
    ├── baseline_{llm}.npy                                                  # cosine baseline scores
    └── aggregation/
        └── flagged_llm={llm}_embedding={emb}_scorer={scorer}_reranker={reranker}_m={m}.parquet
```

---

## File naming convention

Parameters encoded in filenames:

| Token | Possible values | Description |
|---|---|---|
| `llm` | `deepseek`, `gpt`, `gemini` | LLM used for generation |
| `emb` | `gemma`, `bertweet`, `bge-m3` | Embedding model used for retrieval |
| `scorer` | `none`, `roberta_large`, `bertweet`, `bertweet_large`, `bleurt` | BERTScore backbone or BLEURT; `none` for cosine baseline |
| `reranker` | `none`, `bertscore`, `bleurt` | Reranking method applied after cosine retrieval |
| `m` | integer | Number of retrieval candidates per query (top-*m*); `all` for the cosine baseline |

---

## File types

### 1. Retrieval pairs — `results_*.parquet`

Intermediate output of the evaluation stage. Contains one row per (LLM query, human post candidate) pair, for the top-*m* human posts retrieved per query.

| Column | Type | Description |
|---|---|---|
| `llm_row_idx` | int | Row index into the LLM posts DataFrame (from `load_llm_generations`) |
| `human_row_idx` | int | Row index into the human posts DataFrame (from `load_human_posts`) |
| `rank_cosine` | int | Rank of this human post for the given LLM query, by cosine similarity (1 = most similar) |
| `cosine_score` | float | Cosine similarity between the LLM query embedding and the human post embedding |
| `bertscore_score` | float | BERTScore F1 between the LLM query text and the human post text (only present when `scorer != none`) |
| `rank_bertscore` | int | Rank by BERTScore within the top-*m* candidates for this query (only present when `scorer != none`) |

For BLEURT runs, `bertscore_score` and `rank_bertscore` are replaced by `bleurt_score` and `rank_bleurt`.

### 2. Flagged / ranked human posts — `aggregation/flagged_*.parquet`

Final output of the aggregation stage. Contains one row per human post, ranked by its best score across all LLM queries. This is the file used in the notebooks to compute Precision@K and other retrieval metrics.

| Column | Type | Description |
|---|---|---|
| `human_row_idx` | int | Row index into the human posts DataFrame |
| `llm_row_idx` | int | Index of the LLM query that produced the best score for this human post |
| `best_{score}_score` | float | Best score this post received across all LLM queries (e.g. `best_cosine_score`, `best_bertscore_score`) |
| `human_text` | str | Text of the human post |
| `llm_text` | str | Text of the matching LLM-generated post |
| `label` | str | Ground-truth label: `TRUE`, `FALSE`, or `OTHER` |
| `rank` | int | Final rank (1 = most likely to be disinformation according to the model) |

Posts are sorted by `best_{score}_score` descending, so rank 1 is the post the model considers most suspicious.

### 3. BM25 baseline arrays — `baseline_{llm}.npy`

2-D NumPy array of shape (Q × H), where Q is the number of LLM-generated queries and H is the number of human posts. Each entry is the BM25 score assigned to a human post for a given LLM query, computed by `bm25_retriever.py`. Used in the notebooks to plot the lexical baseline precision curves alongside the semantic retrieval results.

---

## Computing evaluation metrics

Load a flagged parquet file and slice by rank to compute Precision@K:

```python
import pyarrow.parquet as pq

df = pq.read_table(
    "results/dataset_1/aggregation/"
    "flagged_llm=deepseek_embedding=gemma_scorer=roberta_large_reranker=bertscore_m=200.parquet"
).to_pandas()

K = 100
top_k = df[df["rank"] <= K]
precision_at_k = (top_k["label"] == "FALSE").sum() / K
print(f"Precision@{K}: {precision_at_k:.3f}")
```

The notebooks in `notebooks/02a_*` and `notebooks/02b_*` load these files and compute the full set of metrics reported in the thesis.
