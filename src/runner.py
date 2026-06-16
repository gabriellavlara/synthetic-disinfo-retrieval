import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
from huggingface_hub import login, whoami
import yaml
import json
from pathlib import Path

import evaluate
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:                                     
    for gpu in gpus:                                                                                          
        tf.config.experimental.set_memory_growth(gpu, True)


from bert_score import BERTScorer # get GPU

from dotenv import load_dotenv
load_dotenv()

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

from utils import load_news_articles, load_prompts, load_llm_generations, load_human_posts
from utils import stage_header, stage_footer
from generation import get_api_client,  run_llm
from embedding import embed_posts
from evaluation import run_similarity_experiment, run_cosine_baseline
from aggregation import flag_human_posts


config_pipeline = config["Pipeline"]
RUN_GENERATION = config_pipeline["run_generation"]
RUN_EMBEDDING = config_pipeline["run_embedding"]
RUN_EVALUATION = config_pipeline["run_evaluation"]
RUN_AGGREGATION = config_pipeline["run_aggregation"]

config_run = config["Run"]
dataset_nr = config_run["dataset_nr"]
MAX_TOKENS = config_run["max_tokens"]
N_REPETITIONS = config_run["n_repetitions"]
llm_models = config_run["llm_models"]
embedding_models = config_run["embedding_models"]

config_evaluation = config["Evaluation"]
config_aggregation = config["Aggregation"]

# Get reranker metadata from config — needed by both evaluation and aggregation blocks
reranker = config_run["reranker"]   # "none" | "bertscore" | "bleurt"
scorer_name = "none"
batch_size = 32
if reranker == "bertscore":
    _scorer_cfg = config_evaluation["reranking"][config_evaluation["active_scorer"]]
    scorer_name = _scorer_cfg["model_name"]
    batch_size  = _scorer_cfg["batch_size"]
elif reranker == "bleurt":
    _bleurt_cfg = config_evaluation["reranking"]["bleurt"]
    scorer_name = _bleurt_cfg["model_name"]
    batch_size  = _bleurt_cfg["batch_size"]

#*-----------------------------------------------------------------------------*
# 1: Load inputs

news = load_news_articles(dataset_nr=dataset_nr)
prompts = load_prompts(file_name="prompts_misinformation_constraints_german.json")
human_posts = load_human_posts(dataset_nr=dataset_nr)

#*-----------------------------------------------------------------------------*
# 2. For each LLM model x news article x prompt, generate an instance
if RUN_GENERATION:
    stage_header("GENERATION", dataset_nr)
    stage_start = time.time()

    for llm_model in llm_models:
        print(f"-> Language model: {llm_model}")

        client = get_api_client(model_name = llm_model) 

        # Crash safe logic
        gen_file =  Path(f"data/generations/dataset_{dataset_nr}/generations_{llm_model}.jsonl")
        completed = set()
        if gen_file.exists():
            with gen_file.open() as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        completed.add((r["articleID"], r["promptID"], r.get("rep_index", 0)))

        for news_article in news:
            for prompt in prompts:
                for rep in range(N_REPETITIONS):
                    key = (news_article["articleID"], prompt["promptID"], rep)
                    if key in completed:
                        continue   # already done, skip

                    result = run_llm(
                            client = client,
                            model_name = llm_model, 
                            news_article = news_article,
                            prompt= prompt,
                            max_tokens = MAX_TOKENS,
                            rep_index = rep
                            )
                    time.sleep(6)
                    
    stage_footer("GENERATION", stage_start)

#*-----------------------------------------------------------------------------*
# 3. For each embedding model, embed synthetic and organic posts
if RUN_EMBEDDING:
    stage_header("EMBEDDING", dataset_nr)
    stage_start = time.time()

    for embedding_model in embedding_models:

        print(f"-> Embedding model: {embedding_model}")

        embed_posts(posts_df= human_posts,
                model_name = embedding_model,
                source = "human",
                dataset_nr=dataset_nr)
        
        for llm_model in llm_models:
            llm_posts = load_llm_generations(dataset_nr=dataset_nr,
                                             model_name=llm_model)
            embed_posts(posts_df = llm_posts, 
                    model_name=embedding_model,
                    source=llm_model,
                    dataset_nr=dataset_nr)  
            
    stage_footer("EMBEDDING", stage_start)


#*-----------------------------------------------------------------------------*
# 4. Retrieval + reranking (skipped for cosine baseline — handled in aggregation)
if RUN_EVALUATION and reranker != "none":
    stage_header("EVALUATION", dataset_nr)
    stage_start = time.time()

    scorer = None
    if reranker == "bertscore":
        scorer_cfg = config_evaluation["reranking"][config_evaluation["active_scorer"]]
        scorer = BERTScorer(
            model_type=scorer_cfg["model_type"],
            num_layers=scorer_cfg["num_layers"],
            device=scorer_cfg["device"]
        )
        if "max_token_length" in scorer_cfg:
            scorer._tokenizer.model_max_length = scorer_cfg["max_token_length"]
    elif reranker == "bleurt":
        bleurt_cfg = config_evaluation["reranking"]["bleurt"]
        scorer = evaluate.load(bleurt_cfg["model_name"], bleurt_cfg["model_checkpoint"])

    print(f"-> Reranker: {reranker} (scorer: {scorer_name})")

    for llm_model in llm_models:
        for embedding_model in embedding_models:
            print(f"-> Language model: {llm_model}")
            print(f"-> Embedding model: {embedding_model}")

            llm_posts = load_llm_generations(dataset_nr=dataset_nr,
                                             model_name=llm_model)

            run_similarity_experiment(
                dataset_nr=dataset_nr,
                llm_model_name=llm_model,
                embedding_model_name=embedding_model,
                human_posts=human_posts,
                llm_posts=llm_posts,
                reranker=reranker,
                scorer=scorer,
                scorer_name=scorer_name,
                overwrite=config_pipeline["overwrite"]["evaluation"],
                m=config_evaluation["retrieval"]["m"],
                batch_size=batch_size,
            )

    stage_footer("EVALUATION", stage_start)

#*-----------------------------------------------------------------------------*
# 5. Flag human posts
if RUN_AGGREGATION:
    stage_header("AGGREGATION", dataset_nr)
    stage_start = time.time()

    for llm_model in llm_models:
        for embedding_model in embedding_models:
            llm_posts = load_llm_generations(dataset_nr=dataset_nr,
                                             model_name=llm_model)
            if reranker == "none":
                # Cosine baseline: rank all H posts directly from the full (Q, H) matrix
                run_cosine_baseline(
                    dataset_nr=dataset_nr,
                    llm_model_name=llm_model,
                    embedding_model_name=embedding_model,
                    human_posts=human_posts,
                    llm_posts = llm_posts,
                    overwrite=config_pipeline["overwrite"]["aggregation"],
                )
            else:
                # Reranking pipeline: read pairs_df, rank by reranker score
                flag_human_posts(
                    dataset_nr=dataset_nr,
                    llm_model_name=llm_model,
                    embedding_model_name=embedding_model,
                    scorer_name=scorer_name,
                    reranker=reranker,
                    human_posts=human_posts,
                    llm_posts = llm_posts,
                    m=config_evaluation["retrieval"]["m"],
                    overwrite=config_pipeline["overwrite"]["aggregation"],
                )

    stage_footer("AGGREGATION", stage_start)
