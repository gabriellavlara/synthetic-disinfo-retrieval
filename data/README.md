# Data

This directory contains all inputs and intermediate artefacts for the pipeline. Large files (embeddings, generations, human-authored corpora) are excluded from version control via `.gitignore`.

```
data/
├── datasets/               # human-authored social media corpora
│   ├── monkeypox-data/     # Dataset 1 — exploratory
│   └── generalization-data/# Datasets 2 & 3 — generalization
├── newsarticles/           # source news articles used to ground generation
├── generations/            # LLM-generated synthetic posts (JSONL)
├── embeddings/             # cached embedding vectors (NumPy)
├── annotations/            # human and LLM-as-judge annotation files
├── prompts_placeholder.json        # Placeholder file for prompts, which are omitted for ethical concerns. 
```

---

## datasets/

### Dataset 1 — `monkeypox-data/monkeypox.csv`

Source: [Monkeypox Tweets Dataset (Kaggle)](https://www.kaggle.com/datasets/stephencrone/monkeypox)

5,787 rows × 19 columns. The pipeline filters to Twitter-origin posts (`source` column) and deduplicates on `text`, yielding 4567 posts.

| Column | Type | Description |
|---|---|---|
| `number` | int | Row identifier |
| `created_at` | datetime | Tweet timestamp |
| `text` | str | Raw tweet text |
| `source` | str | Twitter client used to post (e.g. "Twitter for iPhone") |
| `retweet_count` | int | Retweet count at collection time |
| `like_count` | int | Like count at collection time |
| `binary_class` | int | Binary label: `0` = not misinformation, `1` = misinformation |
| `ternary_class` | int | Ternary label: `0` = TRUE, `1` = FALSE, `9` = OTHER (mixed/satire) |

The pipeline uses `ternary_class` for evaluation. Posts with `ternary_class == 9` are retained and mapped to the OTHER category.

### Datasets 2 & 3 — `generalization-data/`

Source: The posts were curated by the Quality and Usability Lab from the Technical University of Berlin. FOr access, contact [Quality and Usability Lab](https://www.tu.berlin/en/qu)

**`english_posts_llm_experiment.csv`** — 249 rows × 2 columns  
**`german_posts_llm_experiment.csv`** — similar structure

| Column | Type | Description |
|---|---|---|
| `item_raw_content` | str | Raw social media post text (X / Reddit) |
| `label` | str | Fact-check verdict: `TRUE`, `PARTLY_TRUE`, `UNPROVEN`, `MOSTLY_FALSE`, `FALSE` |

The pipeline remaps these to three classes: `TRUE`, `FALSE`, and `OTHER` (covers PARTLY_TRUE and UNPROVEN). URLs are normalised to the `_URL_` token for consistency with Dataset 1.

---

## newsarticles/news-articles.json

5 news articles per topical cluster are used to ground the synthetic post generation. Each article is a JSON object:

| Field | Type | Description |
|---|---|---|
| `articleID` | int | Unique article identifier |
| `eventID` | int | Maps to dataset: `1` = Monkeypox, `2` = English generalization, `3` = German generalization |
| `title` | str | Article title |
| `headline` | str | One-sentence headline / lead |
| `publication_date` | str | ISO 8601 date (YYYY-MM-DD) |
| `url` | str | Source URL |

---

## generations/

Synthetic posts produced by each LLM, one JSONL file per model per dataset.

```
generations/
├── dataset_1/
│   ├── generations_deepseek.jsonl
│   ├── generations_gemini.jsonl
│   └── generations_gpt.jsonl
├── dataset_2/
│   └── generations_deepseek.jsonl
└── dataset_3/
    └── generations_deepseek.jsonl
```

Each line is a JSON object:

| Field | Type | Description |
|---|---|---|
| `genID` | str | UUID for this generation instance |
| `articleID` | int | Source article (foreign key → `news-articles.json`) |
| `modelID` | str | LLM name: `"deepseek"`, `"gpt"`, or `"gemini"` |
| `promptID` | int | Prompt used (foreign key → prompts file) |
| `rep_index` | int | Repetition index (0-based); multiple posts are generated per article × prompt pair |
| `generated_text` | str | The synthetic post text |
| `generation_time` | float | API response time in seconds |
| `generation_cost` | float | Estimated API cost in USD |

The pipeline filters to `promptID` values 6, 7, 9, and 10 (misinformation-constrained prompts) during loading.

---

## embeddings/

Cached embedding vectors in NumPy `.npy` format. Not tracked by git; recomputed by running the embedding stage.

```
embeddings/
└── dataset_{1,2,3}/
    └── {gemma,bertweet,bge-m3}/
        ├── embeddings_human.npy
        ├── embeddings_deepseek.npy
        ├── embeddings_gemini.npy    # dataset_1 only
        └── embeddings_gpt.npy       # dataset_1 only
```

Each file is a 2-D float32 array of shape `(N, d)`, where `N` is the number of posts and `d` is the embedding dimension:

| Model | `d` |
|---|---|
| `gemma` (google/embeddinggemma-300M) | 768 |
| `bertweet` (vinai/bertweet-base) | 768 |
| `bge-m3` (BAAI/bge-m3) | 1024 |

Row order matches the DataFrame returned by `load_human_posts()` or `load_llm_generations()` for the corresponding dataset and model.

---

## annotations/

Human and LLM-as-judge annotation files used in the qualitative analysis (notebook `03_qualitative_results.ipynb`).

| File | Description |
|---|---|
| `top_matches_annotations_merged.csv` | Human inter-annotator study: top retrieval matches labelled by two annotators (Dataset 1) |
| `generalization/llm_as_a_judge_experiment.json` | Input pairs for the LLM-as-judge evaluation (Datasets 2 & 3) |
| `generalization/llm_as_a_judge_experiment_annotated_claude.json` | Claude judgements |
| `generalization/llm_as_a_judge_experiment_annotated_gpt.json` | GPT-4 judgements |

---

## Prompt files

Two JSON files containing the prompts used to instruct the LLMs. Each file is a list of prompt objects:

| Field | Type | Description |
|---|---|---|
| `promptID` | int | Unique prompt identifier |
| `system_prompt` | str | System-role instruction defining the persona |
| `user_prompt` | str | Task instruction with `{{article_title}}` and `{{article_headline}}` placeholders |
| `instruction_technique` | str | Prompting technique (e.g. `"persona"`) |
| `instruction_style` | str | Rhetorical style (e.g. `"politically_biased"`) |
| `task_type` | str | Generation task type (e.g. `"tweet_generation"`) |
| `has_misinformation_constraint` | bool | Whether the prompt requires fabricated content |
| `misinformation_type` | str | Type of misinformation targeted (e.g. `"fabrication"`) |
| `n_shots` | int | Number of few-shot examples (0 for all prompts in this study) |

`prompts_misinformation_constraints.json` is used for Datasets 1 and 2. `prompts_misinformation_constraints_german.json` is the German-language equivalent used for Dataset 3.
