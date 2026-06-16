"""
This file contains all helper functions to the embedding step in the pipeline. 
It includes:
- create_posts_df
- embed_posts
- generate_umap
"""
import json
from pathlib import Path
from tqdm.auto import tqdm
import os
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"  
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from sklearn.preprocessing import normalize
import umap
import matplotlib.pyplot as plt


def create_posts_df (
        human_posts: pd.DataFrame,
        llm_posts: pd.DataFrame,
        dataset_nr: int = 1
        ):
    """
    Creates a unified DF for storing both human- and machine-generated posts

    Parameters
    ----------
    - human_posts: a DF that contains human posts
    - llm_posts: a DF that contains LLM generated posts
    - dataset_nr:
        - 1 for Monkeypox Dataset
        - 2 for English generalization dataset
        - 3 for German generalization dataset

    Returns
    ----------
    - posts: a DF with one row per generated post.
    Contains metainformation like {'post_id', 'text', 'source', 'veracity', 'group'}
    
    """
    
    llm_temp= pd.DataFrame()
    llm_temp["genID"] = llm_posts["genID"]
    llm_temp["promptID"] = llm_posts["promptID"]
    llm_temp["llm_model"] = llm_posts["modelID"]
    llm_temp["newsID"] = llm_posts["articleID"]
    llm_temp['text'] = llm_posts["generated_text"]
    llm_temp['source'] = "LLM"
    llm_temp["veracity"] = np.where(llm_posts["promptID"].isin([1, 2]), "TRUE", "FALSE") 

    # mask = llm_posts['promptID'].isin([1, 2])
    # llm_temp['veracity'] = mask # veracity label false for promptIDs = {3,4,5}

    human_temp = pd.DataFrame()
    human_temp['text'] = human_posts["text"]
    human_temp['source'] = "HUMAN"

    if dataset_nr == 1:
        human_temp["veracity"] = human_posts["ternary_class"].map({0: "TRUE", 1: "FALSE", 9: "OTHER"}).fillna("OTHER")
        human_temp["orig_post_id"] = human_posts["number"]

    elif dataset_nr in (2, 3):
        human_temp["veracity"] = human_posts["label"]
        human_temp["orig_post_id"] = human_posts.index


    posts = pd.concat([llm_temp, human_temp], ignore_index=True)
    posts['post_id'] = np.arange(len(posts))
    posts['datasetID'] = dataset_nr
    posts['group'] = posts['source'] + "_" + posts['veracity']
   

    return posts


def embed_posts (
    posts_df: pd.DataFrame,
    model_name: str,
    batch_size: int = 2048,
    out_dir: str | Path = "data/embeddings",
    source: str = "human",
    normalize: bool = True,
    resume: bool = True,
    overwrite: bool = False,
    dataset_nr: int = 2
) :
    """
    Embed posts_df[text_col] in batches and store results incrementally
    in a single .npy file using a memmap.

    Crash-safe:
      - embeddings are written batch-by-batch
      - progress is tracked in a small JSON file
      - rerunning resumes from last completed batch

    Returns:
      Path to the final .npy embeddings file.
    """
    if len(posts_df) == 0:
        raise ValueError("Empty posts DF!")
    
    # -----------------------------
    # Model set up
    # -----------------------------
    model_config = None

    for emb_cfg in config["Embeddings"]:
        if emb_cfg["model_name"] == model_name:
            model_config = emb_cfg
            break

    if model_config is None:
        raise ValueError(f"Unsupported model name {model_name}")

    modelid = model_config["model_id"]
    model_dim = model_config["model_dim"]
    trust_remote_code = model_config.get("trust_remote_code", False)

    model = SentenceTransformer(modelid, trust_remote_code=trust_remote_code)

    max_seq_length = model_config.get("max_seq_length", None)
    if max_seq_length is not None:
        model.max_seq_length = max_seq_length

    # -----------------------------
    # Paths
    # -----------------------------
    out_dir = Path(out_dir) / f"dataset_{dataset_nr}" / f"{model_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path = out_dir /f"embeddings_{source}.npy"
    prog_path = out_dir / f"embeddings_{source}.progress.json"

    # -----------------------------
    # Overwrite logic
    # -----------------------------
    if overwrite:
        if emb_path.exists():
            emb_path.unlink()
        if prog_path.exists():
            prog_path.unlink()

    if overwrite:
        resume = False

    N = len(posts_df)

    #-----------------------------
    # Resume logic
    # -----------------------------
    start_idx = 0
    if resume and prog_path.exists() and emb_path.exists():
        try:
            prog = json.loads(prog_path.read_text())
            start_idx = int(prog.get("next_index", 0))
            start_idx = max(0, min(start_idx, N))
        except Exception:
            start_idx = 0
    # -----------------------------
    # Create or open memmap
    # -----------------------------
    if not emb_path.exists() or start_idx == 0:
        mmap = np.lib.format.open_memmap(
            emb_path,
            mode="w+",
            dtype=np.float32,
            shape=(N, model_dim),
        )
        start_idx = 0
    else:
        mmap = np.lib.format.open_memmap(
            emb_path,
            mode="r+",
            dtype=np.float32,
        )
        if mmap.shape != (N, model_dim):
            raise ValueError(
                f"Existing embeddings shape {mmap.shape} "
                f"does not match expected {(N, model_dim)}."
            )

    # -----------------------------
    # Batch embedding loop
    # -----------------------------
    total = N - start_idx
    pbar = tqdm(total=total, desc=f"Embedding ({model_name})", unit="rows")

    for lb in range(start_idx, N, batch_size):
        ub = min(lb + batch_size, N)

        if source == 'human':
            col_name = "text"
        else:
            col_name = "generated_text"
        texts = (
            posts_df.iloc[lb:ub][col_name]
            .astype(str)
            .tolist()
        )

        batch = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        ).astype(np.float32)

        if batch.shape != (ub - lb, model_dim):
            raise RuntimeError(
                f"Unexpected embedding shape {batch.shape}, "
                f"expected {(ub - lb, model_dim)}"
            )

        # 1) write embeddings to disk
        mmap[lb:ub, :] = batch
        mmap.flush()

        # 2) only then mark progress as completed
        prog_path.write_text(json.dumps({"next_index": ub}, indent=2))

        # 3) update progress bar
        pbar.update(ub - lb)

    pbar.close()

    # -----------------------------
    # Final sanity check + cleanup
    # -----------------------------
    final = np.load(emb_path, mmap_mode="r")
    assert final.shape == (N, model_dim), "Final embedding shape mismatch!"

    if prog_path.exists():
        prog_path.unlink()

def generate_umap (
                    model_name: str,
                    posts_df: pd.DataFrame,
                    file_dir: str | Path = "data/embeddings/",
                    n_neighbors: int = 15,
                    min_dist: float = 0.1,
                    metric: str = "cosine",
                    dataset_nr: int = 2
    ):
    """
    Generate a 2D UMAP for a low-dimensional representation of the embedding vectors.

    -----------------------------------------
    Params:
    - model_name: Name of the embedding model
    - posts_df: DF containing all posts (synthetic and organic)
    - file_dir: directory to where embeddings file is located
    - show: if True, plots the result

    -----------------------------------------
    Returns:
    - X_umap: matrix containing lower-dim embedding vectors
    
    """
    # Load file
    file_dir = Path(file_dir) / f"dataset_{dataset_nr}"
    file_path = file_dir / f"embeddings_{model_name}.npy"

    if not file_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {file_path}")

    embedding_vecs = np.load(file_path, allow_pickle = False)

    # Sanity check
    if len(posts_df) != embedding_vecs.shape[0]:
        raise ValueError(
            f"Row mismatch: posts_df has {len(posts_df)} rows, "
            f"embeddings has {embedding_vecs.shape[0]} rows. "
        )

    # Create UMAP
    X = normalize(embedding_vecs)
    reducer = umap.UMAP(
        n_neighbors= n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric=metric,
        random_state=42
    )

    X_umap = reducer.fit_transform(X)

    # Plot 
    plots_dir = Path("results/plots") / f"dataset_{dataset_nr}"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fname = plots_dir / f"umap_{model_name}_nn{n_neighbors}.png"

    plt.figure(figsize=(7,6))
    for g in posts_df["group"].unique():
        idx = posts_df["group"] == g

        if g == "LLM_FALSE":
            plt.scatter(
            X_umap[idx, 0],
            X_umap[idx, 1],
            label=g,
            marker="x",     # X marker
            s=80,           # bigger than others
            linewidths=1.5, # makes the X more visible
            alpha=0.9,
            zorder=3
            )
        elif (g == "HUMAN_OTHER" or g == "LLM_TRUE"):
            plt.scatter(
                X_umap[idx, 0],
                X_umap[idx, 1],
                label=g,
                alpha=0.0,
                s=5,
                zorder=1
            )
        elif g == "HUMAN_TRUE":
            plt.scatter(
                X_umap[idx, 0],
                X_umap[idx, 1],
                label=g,
                alpha=0.6,
                s=5,
                zorder=1,
                color = "green"
            )
        elif g == "HUMAN_FALSE":
             plt.scatter(
                X_umap[idx, 0],
                X_umap[idx, 1],
                label=g,
                alpha=0.6,
                s=5,
                zorder=1,
                color = "red"
            )

    plt.legend()
    plt.title(f"UMAP of tweet embeddings using {model_name}")
    plt.savefig(fname)
    plt.close()
    
    return X_umap, reducer