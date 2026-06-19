# Setup Guide

## Requirements

- Python 3.10 or later
- A GPU is required if you want to run the reranking stage (BERTScore / BLEURT). Generation and embedding can run on CPU or Apple Silicon (MPS), but will be slower.

## 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd synthetic-disinfo-retrieval

python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

If you plan to use the BLEURT reranker, TensorFlow is required and will be installed with the above command. If you only need the BERTScore reranker or the cosine baseline, you can comment out `tensorflow` and `evaluate` in `requirements.txt` to reduce installation time.

## 3. Configure API keys

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Open `.env` and set the keys for the LLMs you intend to use:

| Key | Required for |
|---|---|
| `OPENAI_API_KEY` | GPT-4o-mini generation |
| `DEEPSEEK_API_KEY` | DeepSeek generation |
| `GEMINI_API_KEY` | Gemini generation |
| `HUGGINGFACE_API_KEY` | Downloading gated HuggingFace models (optional) |

SSH fields (`JUMPHOST_*`, `TURBO_*`) are only needed if you are running embedding or reranking on a remote GPU server.

## 4. Obtain the datasets

The human-authored corpora are not redistributed in this repository. See `data/README.md` for sources and instructions on obtaining each dataset and where to place the files.

## 5. Configure the run

All pipeline settings are controlled by `config.yaml` in the repository root:

- `Run.dataset_nr` — which dataset to process (1, 2, or 3)
- `Run.llm_models` — list of LLMs to use for generation, e.g. `["deepseek", "gpt"]`
- `Run.embedding_models` — list of embedding models, e.g. `["gemma", "bertweet", "bge-m3"]`
- `Run.reranker` — `none` (cosine baseline), `bertscore`, or `bleurt`
- `Pipeline.run_*` flags — toggle individual stages on or off

Set `Pipeline.run_generation: false` if you want to skip generation and use the pre-generated JSONL files already in `data/generations/`.

## 6. Run the pipeline

Run from the repository root with `src/` on the Python path:

```bash
PYTHONPATH=src python src/runner.py
```

The pipeline executes up to five stages in order:

| Stage | What it does |
|---|---|
| **Generation** | Calls the LLM API to produce synthetic posts; writes JSONL to `data/generations/` |
| **Embedding** | Encodes posts with the chosen embedding model; writes `.npy` to `data/embeddings/` |
| **Evaluation** | Retrieves top-*m* human posts per synthetic query via cosine similarity, then optionally reranks; writes Parquet to `results/dataset_{nr}/` |
| **Aggregation** | Collapses retrieval results into a single ranked list of human posts; writes Parquet to `results/dataset_{nr}/aggregation/` |

Results are saved incrementally. If a run is interrupted, re-running with `overwrite: false` in `config.yaml` will resume from where it stopped.

## 7. Explore results

Open the notebooks in order:

```
notebooks/
├── 01a_exploratory_dataset.ipynb       # Dataset 1 (Monkeypox) exploration
├── 01b_generalization_dataset.ipynb    # Datasets 2 & 3 exploration
├── 02a_results_exploratory_dataset.ipynb
├── 02b_results_generalization_dataset.ipynb
└── 03_qualitative_results.ipynb
```

The `02*` and `03` notebooks load results from `results/` and `data/annotations/` and reproduce the figures and tables in the thesis.

## Troubleshooting

**`ModuleNotFoundError: No module named 'utils'`** — Make sure you are running from the repository root with `PYTHONPATH=src`.

**`FileNotFoundError: config.yaml`** — Same cause; `config.yaml` is read with a relative path and the working directory must be the repository root.

**Out-of-memory during reranking** — Reduce `batch_size` in `config.yaml` under `Evaluation.reranking.roberta` (default 16) or switch to `bertweet` (default 32, smaller model).

**Apple Silicon (MPS) and reranking** — BERTScore and BLEURT are not reliably supported on MPS. Set `device: cpu` under the relevant reranker in `config.yaml`, or run reranking on a CUDA machine.
