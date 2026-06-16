"""
Helper functions that will be used across the files.

This utils.py contains:
- load_news_articles
- load_human_posts
- load_llm_generations
- load_prompts

- stage_header
- stage_footer
"""
import os
import pandas as pd
import numpy as np
from huggingface_hub import login, whoami
import pyarrow
from pathlib import Path
import json
import time
from datetime import datetime
import pyarrow.parquet as pq 
import re
__all__ = ['load_news_articles', 'load_prompts', 'load_llm_generations', 'load_human_posts', 'load_experiment_results']
#------------------------------------------------------
# LOAD FUNCTIONS
#------------------------------------------------------
def load_news_articles(
    base_dir: str | Path = "data/newsarticles/",
    file_name: str = "news-articles.json",
    as_df: bool = False,
    dataset_nr: int = 1
):
    """
    Loads news articles stored as JSON into memory.

    Parameters
    ----------
    base_dir : Path to news data directory.
    file_name : JSON filename (expects a list of article objects).
    as_df : If True, return a pandas DataFrame; otherwise return a list[dict]

    Returns
    -------
    If as_df=True:
        pd.DataFrame with one row per article.
    If as_df=False:
        list[dict] where each dict has keys like:
        articleID, eventID, title, headline, publication_date, url
    """
    base_dir = Path(base_dir)
    file_path = base_dir / file_name

    with open(file_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    # Basic schema check (fail early)
    required = {"articleID", "eventID", "title", "headline", "publication_date", "url"}
    for i, a in enumerate(articles):
        if not isinstance(a, dict):
            raise ValueError(f"Article at position {i} is not an object/dict.")
        missing = required - set(a.keys())
        if missing:
            raise ValueError(f"Article at position {i} is missing keys: {sorted(missing)}")

    if dataset_nr not in (1, 2, 3):
        raise ValueError(f"dataset_nr must be 1–3, got {dataset_nr}")
    articles = [a for a in articles if a["eventID"] == dataset_nr]
        
    if not as_df: 
        return articles 
    else: 
        news_df = pd.DataFrame(articles)
        news_df["publication_date"] = pd.to_datetime(
            news_df["publication_date"], errors="raise"
        )
        return news_df

def load_prompts (
        base_dir: str | Path = "data/",
        file_name: str = "prompts.json",
        as_df: bool = False
):
    """
    Loads prompts stored in JSON file. 

    Parameters
    ----------
    base_dir : Path to JSON file.
    file_name : File name
    as_df : If True, return a pandas DataFrame; otherwise return a list[dict]

    If as_df=True:
        pd.DataFrame with one row per article.
    If as_df=False:
        list[dict] where each dict has keys like:
        promptID, system_prompt, user_prompt, instruction_technique, instruction_style, task_type, n_shots
    """
    base_dir = Path(base_dir)
    file_path = base_dir / file_name

    with open(file_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    # Basic schema check (fail early)
    required = required = {"promptID","system_prompt","user_prompt","instruction_technique","instruction_style","task_type","n_shots"}

    for i, p in enumerate(prompts):
        if not isinstance(p, dict):
            raise ValueError(f"Prompt at position {i} is not an object/dict.")
        missing = required - set(p.keys())
        if missing:
            raise ValueError(f"Prompt at position {i} is missing keys: {sorted(missing)}")

    if not as_df: 
        return prompts
    else: 
        prompts_df = pd.DataFrame(prompts)
        return prompts_df


def load_llm_generations(
    dataset_nr: int, 
    model_name: str,
    as_df: bool = True,
):
    """
    Loads generated posts stored in a JSONL file (one JSON object per line).

    Parameters
    ----------
    base_dir : Path to directory containing the JSONL file
    file_name : JSONL file name
    as_df : If True, return a pandas DataFrame; otherwise return a list[dict]

    Returns
    -------
    If as_df=True:
        pd.DataFrame with one row per generated instance.
    If as_df=False:
        list[dict] where each dict has keys like:
        "genID", "articleID", "modelID", "promptID", "generated_text"
        (and optionally "viewpoint", etc.)
    """

    base_dir = Path(f"data/generations/dataset_{dataset_nr}")
    file_path = base_dir / f"generations_{model_name}.jsonl"

    if not file_path.exists():
        raise FileNotFoundError(f"Generations file not found: {file_path}")

    llm_generations = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue  # skip empty lines
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_idx + 1} of {file_path}"
                ) from e
            llm_generations.append(record)

    # Basic schema check (fail early)
    required = {"genID", "articleID", "modelID", "promptID", "rep_index", "generated_text"}

    for i, rec in enumerate(llm_generations):
        if not isinstance(rec, dict):
            raise ValueError(f"Generation record at position {i} is not a dict.")
        missing = required - set(rec.keys())
        if missing:
            raise ValueError(
                f"Generation record at position {i} is missing keys: {sorted(missing)}"
            )

    if not as_df:
        return llm_generations
  
    df_llm_generations = pd.DataFrame(llm_generations)
    if "rep_index" not in df_llm_generations.columns:
        print("[WARN] rep_index column not found — old generations file, defaulting to 0.")
        df_llm_generations["rep_index"] = 0  
        
    df_llm_generations = df_llm_generations[df_llm_generations["promptID"].isin([6,7,9,10])].reset_index(drop=True)
    return df_llm_generations

def load_human_posts(dataset_nr: int = 1):
    """
    Load human posts based on dataset nr.
    1 = Monkeypox dataset
    2 = English generalization dataset
    3 = German generalization dataset
    """
    def is_mostly_url(text):
            cleaned = str(text)
            cleaned = re.sub(r'http\S+|www\.\S+', '', cleaned)   # remove URLs
            cleaned = re.sub(r'@\w+', '', cleaned)                # remove @mentions
            cleaned = cleaned.strip()
            return len(cleaned) < 20
    def replace_urls(text):
        text = str(text)
    
    # replace literal \n string with actual newline first
        text = text.replace('\\n', '\n')
        
        # markdown links [text](url)
        text = re.sub(r'\[([^\]]+)\]\(https?://\S+\)', r'\1', text)
        
        # http/https URLs — greedy, catches truncated ones too
        text = re.sub(r'https?://[^\s\)\]]+', '_URL_', text)
        
        # www. URLs
        text = re.sub(r'www\.[^\s\)\]]+', '_URL_', text)
        
        # bare domains with path (e.g. timesofisrael.com/labor-chief...)
        text = re.sub(r'\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}/[^\s\)\]]*', '_URL_', text)
        
        # bare domains without path that look like news sites (e.g. nwww, \nwww artifacts)
        text = re.sub(r'\bn?[a-zA-Z0-9.-]+\.(com|org|net|de|uk|io|gov|edu|fr|nl|au|ca|il|us)\b', '_URL_', text)
        
        # clean up trailing artifacts
        text = re.sub(r'_URL_[^\s]*', '_URL_', text)  # catch anything stuck to _URL_
        text = re.sub(r'\(_URL_\)\.?', '_URL_', text)
        
        return text.strip()

    if dataset_nr == 1:
        file_path = "data/datasets/monkeypox-data/monkeypox.csv"
        df_monkeypox = pd.read_csv(file_path)

        twitter_sources = ["Twitter for iPhone", "Twitter Web App","Twitter for Android",
                   "Twitter for iPad", "TweetDeck","Twitter" ]
        df_monkeypox = df_monkeypox[df_monkeypox['source'].isin(twitter_sources)].reset_index(drop=True)

        df = df_monkeypox[['number', 'created_at', 'text',
                           'source','retweet_count', 'like_count',
                           'binary_class', 'ternary_class']]
        
        df = df.drop_duplicates(subset="text").reset_index(drop=True)
        # df = df[df["ternary_class"].isin([0,1])].reset_index(drop=True)

    elif dataset_nr == 2:
        file_path = "data/datasets/generalization-data/english_posts_llm_experiment.csv"
        df_eng = pd.read_csv(file_path)

        df_eng = df_eng.rename(columns={"item_raw_content": "text", "final_normalized": "label"})
        df_eng= df_eng[["text", "label"]]

        label_map = {
            "TRUE": "TRUE",
            "PARTLY_TRUE": "OTHER",
            "UNPROVEN": "OTHER",
            "MOSTLY_FALSE": "FALSE",
            "FALSE": "FALSE",
            "FALSE ": "FALSE"  # trailing space guard
        }

        df_eng["label"] = df_eng["label"].str.strip().map(label_map)
        df_eng["X_or_Reddit"] = df_eng["text"].apply(lambda x: "Reddit" if len(x) > 280 else "X")

        # drop duplicates and remove entries with mostly URLs
        def is_mostly_url(text):
            cleaned = str(text)
            cleaned = re.sub(r'http\S+|www\.\S+', '', cleaned)   # remove URLs
            cleaned = re.sub(r'@\w+', '', cleaned)                # remove @mentions
            cleaned = cleaned.strip()
            return len(cleaned) < 20

        mask = df_eng["text"].apply(is_mostly_url)
        df_eng = df_eng[~mask].reset_index(drop=True)

        # Drop duplicates
        df_eng = df_eng.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)

        # Normalize URLs to _URL_ token for consistency with the exploratory dataset
        df_eng["text"] = df_eng["text"].apply(replace_urls)

        df = df_eng[["text", "label"]]
    elif dataset_nr == 3:
        file_path = "data/datasets/generalization-data/german_posts_llm_experiment.csv"
        df_ger = pd.read_csv(file_path)

        df_ger = df_ger.rename(columns = {"item_raw_content": "text", "final_normalized": "label"})[["text", "label"]]

        label_map = {
                    "TRUE": "TRUE",
                    "PARTLY_TRUE": "OTHER",
                    "UNPROVEN": "OTHER",
                    "MOSTLY_FALSE": "FALSE",
                    "FALSE": "FALSE",
                    "FALSE ": "FALSE"  # trailing space guard
                }

        df_ger["label"] = df_ger["label"].str.strip().map(label_map)
        df_ger["X_or_Reddit"] = df_ger["text"].apply(lambda x: "Reddit" if len(x) > 280 else "X")

        # drop duplicates and remove entries with mostly URLs
       

        mask = df_ger["text"].apply(is_mostly_url)
        df_ger = df_ger[~mask].reset_index(drop=True)

        # Drop duplicates
        df_ger= df_ger.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)

        # Normalize URLs to _URL_ token for consistency with the exploratory dataset
        df_ger["text"] = df_ger["text"].apply(replace_urls)

        df = df_ger[["text", "label"]]

    else:
        raise ValueError("Invalid dataset nr!")
        
    return df

def load_experiment_results (
        dataset_nr,
        llm_name,
        embedding_name,
        scorer_name,
        m,
):
    """
    Load experiment results and original inputs based on a combination of dataset_nr, LLM_name and embedding_name.

    Parameters
    ----------
    m : int or str
        Number of retrieval candidates used when the experiment was run (e.g. 100 or 4567).
        Must match the value used during the run so the correct file is loaded.

    Returns:
    - pairs_df: instance level pairings of (HUMAN)
    - llm_posts: posts generated by LLM
    - human_posts: human generated posts
    """
    filename = f"results/dataset_{dataset_nr}/results_llm={llm_name}_embedding={embedding_name}_scorer={scorer_name}_m={m}.parquet"
    table = pq.read_table(filename)
    pairs_df = table.to_pandas()

    llm_posts = load_llm_generations(dataset_nr=dataset_nr,
                                    model_name=llm_name)

    human_posts = load_human_posts(dataset_nr=dataset_nr)

    return pairs_df, llm_posts, human_posts
#------------------------------------------------------
# STAGE FUNCTIONS 
#------------------------------------------------------
def stage_header(stage_name, dataset_nr):
    print("\n" + "=" * 60)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
    print(f"STAGE: {stage_name}")
    print(f"Dataset: {dataset_nr}")
    print("=" * 60)

def stage_footer(stage_name, start_time):
    duration = time.time() - start_time
    print(f"Finished {stage_name} in {duration:.2f} seconds")
    print("=" * 60 + "\n")



