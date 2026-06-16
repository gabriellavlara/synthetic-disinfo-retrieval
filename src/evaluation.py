"""
This file contains all helper functions to evaluate the results of the experiment. 
It includes:
- _retrieve_candidates_with_cosine_similarity
- _rerank_with_bertscore
- _build_pairs_df
- run_similarity_experiment
- ...
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from bert_score import BERTScorer
import os
from tqdm import tqdm

def _retrieve_candidates_with_cosine_similarity(
                                                dataset_nr: int,
                                                llm_model_name: str,
                                                embedding_model_name: str,
                                                m: int 
                                                ):
    """
    Retrieve M candidates based on cosine similarity between (LLM_FALSE, HUMAN_) post.

    Params:
    - m: number of candidates to retrieve

    Returns:
    - top_idx: indices into HUMAN embedding matrix for top-m candidates per query;  shape (Q,m)
    - top_scores: corresponding cosine similarity scores; shape (Q,m)

    """
    # ****************************************
    # 1. Load embeddings
    # ****************************************
    base_dir = Path(f"data/embeddings/dataset_{dataset_nr}/{embedding_model_name}")
    llm_emb_file_path = base_dir / f"embeddings_{llm_model_name}.npy"
    human_emb_file_path = base_dir / "embeddings_human.npy"
    
    if (not llm_emb_file_path.exists()) or (not human_emb_file_path.exists()):
        raise FileNotFoundError("Embeddings file not found!")

    llm_emb = np.load(llm_emb_file_path) #shape (Q,d)
    human_emb = np.load(human_emb_file_path) #shape (H,d)

    H = human_emb.shape[0]
    if m > H:
        m = H 

    # ****************************************
    # 2. Compute cosine similarity
    # ****************************************
    S = cosine_similarity(
        llm_emb, 
        human_emb
    )  # shape: (Q, H) with Q: #LLM_FALSE and H: #HUMAN 

    # ****************************************
    # 3. Return top M values per query
    # ****************************************
    top_idx_unsorted = np.argpartition(-S, kth=m-1, axis=1)[:, :m]  # (Q, m)
    top_scores_unsorted = np.take_along_axis(S, top_idx_unsorted, axis=1)
    
    order = np.argsort(-top_scores_unsorted, axis=1)
    top_idx = np.take_along_axis(top_idx_unsorted, order, axis=1)
    top_scores = np.take_along_axis(top_scores_unsorted, order, axis=1)

    return top_idx, top_scores


def _rerank_with_bertscore(scorer,
                           top_idx: np.ndarray,
                           human_posts: pd.DataFrame,
                           llm_posts: pd.DataFrame,
                           batch_size=32):
    Q, m = top_idx.shape
    all_F1 = []

    assert top_idx.shape[1] == m, f"Shape mismatch: {top_idx.shape}"
    assert len(human_posts) >= m, f"Not enough human posts: {len(human_posts)} < {m}"

    for llm_row_idx in range(Q):
        llm_text = llm_posts.iloc[llm_row_idx]["generated_text"]
        human_rows = human_posts.iloc[top_idx[llm_row_idx]]
        human_texts = human_rows["text"].tolist()

        references = [llm_text] * m
        q_F1 = [0.0] * m
        indices_to_score = list(range(m))

        texts_to_score = [human_texts[i] for i in indices_to_score]
        refs_to_score = [references[i] for i in indices_to_score]

        scored_F1 = []
        for start in range(0, len(texts_to_score), batch_size):
            cands = texts_to_score[start:start+batch_size]
            refs  = refs_to_score[start:start+batch_size]

            _, _, F1 = scorer.score(cands, refs)
            scored_F1.extend(F1.cpu().tolist())

        for original_idx, score in zip(indices_to_score, scored_F1):
            q_F1[original_idx] = score

        all_F1.extend(q_F1)

    return np.array(all_F1).reshape(Q, m)

def _rerank_with_bleurt(scorer,
                        top_idx: np.ndarray,
                        human_posts: pd.DataFrame,
                        llm_posts: pd.DataFrame,
                        batch_size=32):
    Q, m = top_idx.shape
    all_scores = []

    assert top_idx.shape[1] == m, f"Shape mismatch: {top_idx.shape}"
    assert len(human_posts) >= m, f"Not enough human posts: {len(human_posts)} < {m}"

    for llm_row_idx in tqdm(range(Q), desc="BLEURT reranking"):
        llm_text = llm_posts.iloc[llm_row_idx]["generated_text"]
        human_rows = human_posts.iloc[top_idx[llm_row_idx]]
        human_texts = human_rows["text"].tolist()

        references = [llm_text] * m
        q_scores = [0.0] * m
        indices_to_score = list(range(m))

        texts_to_score = [human_texts[i] for i in indices_to_score]
        refs_to_score = [references[i] for i in indices_to_score]

        scored = []
        for start in range(0, len(texts_to_score), batch_size):
            cands = texts_to_score[start:start+batch_size]
            refs  = refs_to_score[start:start+batch_size]
            results = scorer.compute(predictions=cands, references=refs)["scores"]  # returns a list of floats
            scored.extend(results)

        for original_idx, score in zip(indices_to_score, scored):
            q_scores[original_idx] = score

        all_scores.extend(q_scores)

    return np.array(all_scores).reshape(Q, m)


def _build_pairs_df(
                    top_idx: np.ndarray,
                    top_scores: np.ndarray,
                    rerank_scores: np.ndarray = None,
                    reranker_name: str = None,
                    ) -> pd.DataFrame:
    """
    Build a lightweight pairs DataFrame containing only indices and scores.
    Text can be joined back on demand via llm_row_idx / human_row_idx.

    Params:
    - top_idx:       shape (Q, m) — indices into human_posts
    - top_scores:    shape (Q, m) — cosine similarity scores
    - rerank_scores: shape (Q, m) — reranker scores (optional, None if no reranking)
    - reranker_name: name used for column labels, e.g. "bertscore" or "bleurt"

    Returns:
    - pairs_df: DataFrame with cosine columns always present; reranker columns added if rerank_scores provided
    """
    if top_idx.shape != top_scores.shape:
        raise ValueError(f"Shape mismatch: top_idx {top_idx.shape}, top_scores {top_scores.shape}")
    if rerank_scores is not None and top_idx.shape != rerank_scores.shape:
        raise ValueError(f"Shape mismatch: top_idx {top_idx.shape}, rerank_scores {rerank_scores.shape}")

    Q, m = top_idx.shape

    pairs_df = pd.DataFrame({
        "llm_row_idx":   np.repeat(np.arange(Q), m),
        "human_row_idx": top_idx.flatten(),
        "rank_cosine":   np.tile(np.arange(1, m + 1), Q),
        "cosine_score":  top_scores.flatten(),
    })

    if rerank_scores is not None and reranker_name is not None:
        score_col = f"{reranker_name}_score"
        rank_col  = f"rank_{reranker_name}"
        pairs_df[score_col] = rerank_scores.flatten()
        pairs_df[rank_col] = (
            pairs_df
            .groupby("llm_row_idx")[score_col]
            .rank(method="first", ascending=False)
            .astype(int)
        )

    return pairs_df

def run_similarity_experiment(
                                dataset_nr: int,
                                llm_model_name: str,
                                embedding_model_name: str,
                                human_posts: pd.DataFrame,
                                llm_posts: pd.DataFrame,
                                reranker: str,
                                scorer=None,
                                scorer_name: str = "none",
                                overwrite: bool = False,
                                m: int = 200,
                                batch_size: int = 32,
                                ):
    """
    Run retrieval + optional reranking experiment.

    Params:
    - reranker:    "none" | "bertscore" | "bleurt"
    - scorer:      scorer object matching the reranker (None if reranker="none")
    - scorer_name: used in the output filename

    Returns:
    - path to saved parquet file
    """
    # ****************************************
    # 1. Retrieval
    # ****************************************
    top_idx, top_scores = _retrieve_candidates_with_cosine_similarity(
                                                dataset_nr,
                                                llm_model_name,
                                                embedding_model_name,
                                                m)

    # ****************************************
    # 2. Rerank (skipped if reranker="none")
    # ****************************************
    rerank_scores = None
    if reranker == "bertscore":
        rerank_scores = _rerank_with_bertscore(scorer, top_idx, human_posts, llm_posts, batch_size)
    elif reranker == "bleurt":
        rerank_scores = _rerank_with_bleurt(scorer, top_idx, human_posts, llm_posts, batch_size)

    # ****************************************
    # 3. Build pairs DF
    # ****************************************
    reranker_label = reranker if reranker != "none" else None
    pairs_df = _build_pairs_df(top_idx, top_scores, rerank_scores, reranker_label)

    # ****************************************
    # 4. Save
    # ****************************************
    base_dir = Path(f"results/dataset_{dataset_nr}")
    base_dir.mkdir(parents=True, exist_ok=True)
    results_file_path = base_dir / f"results_llm={llm_model_name}_embedding={embedding_model_name}_scorer={scorer_name}_m={m}.parquet"

    if results_file_path.exists() and not overwrite:
        print(f"Results already exist at {results_file_path}. Skipping (overwrite=False).")
        return results_file_path

    pairs_df.to_parquet(results_file_path, compression="gzip")
    print(f"Results file stored in {results_file_path}")
    return results_file_path
    
def run_cosine_baseline(
                        dataset_nr: int,
                        llm_model_name: str,
                        embedding_model_name: str,
                        human_posts: pd.DataFrame,
                        llm_posts: pd.DataFrame,
                        overwrite: bool,
                        ) -> Path:
    """
    Cosine baseline: rank all H human posts by their max cosine similarity to any LLM query.
    No top-m cutoff — every human post gets a score.
    Saves directly to the aggregation folder in the same ranked format as flag_human_posts.
    """
    base_dir = Path(f"data/embeddings/dataset_{dataset_nr}/{embedding_model_name}")
    llm_emb_path   = base_dir / f"embeddings_{llm_model_name}.npy"
    human_emb_path = base_dir / "embeddings_human.npy"

    if not llm_emb_path.exists() or not human_emb_path.exists():
        raise FileNotFoundError("Embeddings not found.")

    llm_emb   = np.load(llm_emb_path)    # (Q, d)
    human_emb = np.load(human_emb_path)  # (H, d)

    # Full (Q, H) cosine matrix — no top-m cutoff
    S = cosine_similarity(llm_emb, human_emb)  # (Q, H)
    best_cosine = S.max(axis=0)                 # (H,)
    best_llm_idx = S.argmax(axis=0)             # (H,) — which LLM post produced the max score

    base = pd.DataFrame({
        "human_row_idx":     np.arange(len(human_emb)),
        "llm_row_idx":       best_llm_idx,
        "best_cosine_score": best_cosine,
    })

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
            .sort_values("best_cosine_score", ascending=False)
            .reset_index(drop=True)
        )
    elif dataset_nr in (2, 3):
        ranked_df = (
            base
            .merge(human_posts[["text", "label"]], left_on="human_row_idx", right_index=True, how="left")
            .rename(columns={"text": "human_text"})
            .merge(llm_posts[["generated_text"]], left_on="llm_row_idx", right_index=True, how="left")
            .rename(columns={"generated_text": "llm_text"})
            .sort_values("best_cosine_score", ascending=False)
            .reset_index(drop=True)
        )
    else:
        raise ValueError(f"Invalid dataset_nr: {dataset_nr}")

    ranked_df["rank"] = range(1, len(ranked_df) + 1)

    out_dir = Path(f"results/dataset_{dataset_nr}/aggregation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"flagged_llm={llm_model_name}_embedding={embedding_model_name}_reranker=none_m=all.parquet"

    if out_path.exists() and not overwrite:
        print(f"Cosine baseline already exists at {out_path}. Skipping (overwrite=False).")
        return out_path

    ranked_df.to_parquet(out_path, compression="gzip", index=False)
    print(f"Cosine baseline stored in {out_path}")
    return out_path

