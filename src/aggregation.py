"""
This file contains all helper functions to aggregate the results of the experiment into interpretable metrics.

It includes:
- flag_human_posts : ranks all human posts by reranker score and saves a single ranked parquet file.
                     Precision/recall/F1 for any K can be computed on the fly from this file.
"""

from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq


def flag_human_posts(
    dataset_nr: int,
    llm_model_name: str,
    embedding_model_name: str,
    scorer_name: str,
    reranker: str,
    human_posts: pd.DataFrame,
    llm_posts: pd.DataFrame,
    m: int,
    overwrite: bool,
):
    """
    Ranks all human posts by their best reranker score across all LLM queries.
    Saves a single ranked parquet — any K is free via ranked_df[ranked_df["rank"] <= k].

    Score column used:
    - reranker="none"      -> cosine_score
    - reranker="bertscore" -> bertscore_score
    - reranker="bleurt"    -> bleurt_score

    Saves:
    - flagged_{stem}.parquet  — all human posts sorted by score, with a rank column
    """
    # 1. Load pairs_df
    pairs_path = Path(
        f"results/dataset_{dataset_nr}/"
        f"results_llm={llm_model_name}_embedding={embedding_model_name}"
        f"_scorer={scorer_name}_m={m}.parquet"
    )
    if not pairs_path.exists():
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")
    pairs_df = pq.read_table(pairs_path).to_pandas()

    # 2. Determine score column
    score_col = "cosine_score" if reranker == "none" else f"{reranker}_score"
    if score_col not in pairs_df.columns:
        raise ValueError(f"Score column '{score_col}' not found. Available: {list(pairs_df.columns)}")

    # 3. Best score per human post across all LLM queries — aggregate and sort once
    best_col = f"best_{score_col}"

    best_idx = pairs_df.groupby("human_row_idx")[score_col].idxmax()
    base = pairs_df.loc[best_idx, ["human_row_idx", "llm_row_idx", score_col]].rename(columns={score_col: best_col})

    if dataset_nr == 1:
        label_map = {0: "TRUE", 1: "FALSE", 9: "OTHER"}
        ranked_df = (
            base
            .merge(human_posts[["text", "ternary_class"]], left_on="human_row_idx", right_index=True, how="left")
            .assign(label=lambda df: df["ternary_class"].replace(label_map))
            .drop(columns=["ternary_class"])
            .rename(columns={"text": "human_text"})
            .merge(llm_posts[["generated_text"]], left_on="llm_row_idx", right_index=True, how="left")
            .rename(columns={"generated_text": "llm_text"})
            .sort_values(best_col, ascending=False)
            .reset_index(drop=True)
        )
    elif dataset_nr in (2, 3):
        ranked_df = (
            base
            .merge(human_posts[["text", "label"]], left_on="human_row_idx", right_index=True, how="left")
            .rename(columns={"text": "human_text"})
            .merge(llm_posts[["generated_text"]], left_on="llm_row_idx", right_index=True, how="left")
            .rename(columns={"generated_text": "llm_text"})
            .sort_values(best_col, ascending=False)
            .reset_index(drop=True)
        )
    else:
        raise ValueError(f"Invalid dataset_nr: {dataset_nr}")
    
    ranked_df["rank"] = range(1, len(ranked_df) + 1)
    
    # 4. Save
    base_dir = Path(f"results/dataset_{dataset_nr}/aggregation")
    base_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"llm={llm_model_name}_embedding={embedding_model_name}"
        f"_scorer={scorer_name}_reranker={reranker}_m={m}"
    )
    flag_path = base_dir / f"flagged_{stem}.parquet"

    if flag_path.exists() and not overwrite:
        print(f"Flagging results already exist at {flag_path}. Skipping (overwrite=False).")
        return flag_path

    ranked_df.to_parquet(flag_path, compression="gzip", index=False)
    print(f"Ranked human posts stored in {flag_path}")
    return flag_path
