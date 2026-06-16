# Investigating Generative Capacities of LLMs for Emerging Disinformation Detection
## A Proactive Framework for Flagging Disinformation Posts via Semantic Similarity

This repository implements a proof-of-concept framework that uses Large Language Models (LLMs) to generate synthetic disinformation-like posts grounded in real news events, and treats these synthetic posts as semantic retrieval proxies for flagging human-authored false content — without relying on pre-existing labelled corpora. It is intended as a triage tool that surfaces candidate posts for human fact-checkers, not as a standalone classifier.

### Research questions

**RQ1** — Can LLMs be prompted to generate disinformation content that is semantically similar to human-authored disinformation?

**RQ2** — Can synthetic fakes serve as proxies for identifying false social media posts?

### Pipeline overview

![Pipeline overview](figures/pipeline.png)

### Repository structure

````
disinformation-detection/
├── data/
│   ├── datasets/          # human-authored corpora (exploratory + generalization)
│   ├── newsarticles/      # source events used to ground generation
│   ├── generations/       # synthetic posts (DeepSeek, Gemini, GPT)
│   ├── embeddings/        # cached embeddings per retrieval model
│   └── annotations/       # human annotation files
├── notebooks/
│   ├── 01a_exploratory_dataset.ipynb
│   ├── 01b_generalization_dataset.ipynb
│   ├── 02a_results_exploratory_dataset.ipynb
│   └── 02b_results_generalization_dataset.ipynb
├── results/               # metrics output (Precision@K, nDCG@K, similarity scores)
├── src/
│   ├── generation.py      # LLM prompting and synthetic post generation
│   ├── embedding.py       # embedding via Gemma, BGE-M3, BERTweet
│   ├── bm25_retriever.py  # lexical baseline
│   ├── evaluation.py      # retrieval + similarity metrics
│   ├── aggregation.py     # cross-configuration aggregation
│   ├── visualization.py   # figures
│   ├── runner.py          # pipeline entry point
│   └── utils.py
├── .gitignore
├── config.yaml            # run configuration
├── requirements.txt
├── SETUP.md
└── README.md
````

### Setup

See `SETUP.md` for environment setup and `config.yaml` for run configuration. Dependencies are pinned in `requirements.txt`:

````bash
pip install -r requirements.txt
````

### Data availability

The human-authored corpora are not redistributed here. See `data/datasets/` for sources and instructions on obtaining them. <!-- adjust to reality -->

### Thesis

The complete master thesis associated with this pipeline is available here: [link](#).
````


