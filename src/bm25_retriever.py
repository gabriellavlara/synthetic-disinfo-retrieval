import yaml
import time
import numpy as np
import os
from pathlib import Path
import re

from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
load_dotenv()

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

from utils import load_news_articles, load_prompts, load_llm_generations, load_human_posts
from utils import stage_header, stage_footer

def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)        # remove URLs
    text = re.sub(r"@\w+", "", text)           # remove mentions
    text = re.sub(r"[^a-z0-9\s]", "", text)   # remove punctuation
    return text.split()


if __name__ == "__main__":
    # Basic config to load inputs
    config_run = config["Run"]
    dataset_nr = config_run["dataset_nr"]

    stage_header("BASELINE COMPUTATION", dataset_nr)
    stage_start = time.time()

    # Load human posts
    human_posts = load_human_posts(dataset_nr=dataset_nr)
    corpus = human_posts["text"]

    tokenized_corpus  = [tokenize(doc) for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)

    # Load generated instances
    for llm_model in ["deepseek"]:
        llm_posts = load_llm_generations(dataset_nr=dataset_nr,
                                             model_name=llm_model)

        queries = llm_posts["generated_text"]
        tokenized_queries = [tokenize(doc) for doc in queries]

        # Compute BM25 lexical similarity
        S = np.array([bm25.get_scores(query) for query in tokenized_queries])

        # Store baseline results
        np.save(f"results/dataset_{dataset_nr}/baseline_{llm_model}.npy", S)

    stage_footer("BASELINE COMPUTATION", stage_start)


