# DocGuide

**DocGuide: A Graph-Guided Reasoning Framework for Heterogeneous Document QA**

This repository contains the implementation of **DocGuide** for question answering over documents containing tables and text.

## Overview

DocGuide is a graph-guided retrieval and reasoning framework for heterogeneous document question answering. It organizes table records, text passages, and intermediate answers in a shared evidence graph to support reasoning across structured and unstructured evidence.

The framework combines graph-guided decomposition and retrieval, incremental evidence view maintenance, and reasoning with verification. Questions are dynamically decomposed into dependent tasks, while a compact evidence view preserves useful results and their supporting dependencies. Verification checks the resulting reasoning path and enables local repair while retaining valid branches.

## Project Structure

```text
├── dataset/                            # Datasets
│   ├── HybridQA/
│   │   └── dev-*.jsonl
│   ├── OTT-QA/
│   │   └── dev-*.jsonl
│   └── MultimodalQA/
│       └── dev-*.jsonl
├── model/                              # Downloaded model checkpoints
│   ├── embedding/
│   └── reranker/
├── prompt/                             # Method prompt templates
├── scripts/
│   ├── evidence_graph/                 # Evidence graph construction
│   │   ├── graph.py
│   │   └── aggregation.py
│   ├── decomposition_retrieval/        # Task planning and retrieval
│   │   ├── task_dag.py
│   │   ├── router.py
│   │   ├── access.py
│   │   ├── fusion.py
│   │   └── row_templates.py
│   ├── view_maintenance/               # Evidence selection and protection
│   │   ├── view.py
│   │   ├── dependency_protection.py
│   │   ├── ensemble_selection.py
│   │   └── pruning.py
│   ├── reasoning_verification/         # Answering, verification, and repair
│   │   ├── answer.py
│   │   ├── verifier.py
│   │   └── repair.py
│   ├── pipeline.py
│   ├── run.py
│   └── prepare_row_templates.py
├── utils/
│   ├── data/                           # Data loading and evidence references
│   ├── llm/                            # API configuration and requests
│   ├── model/                          # Model loading code
│   │   ├── embedding.py
│   │   └── reranker.py
│   ├── evaluation.py                   # Accuracy evaluation
│   ├── config.py
│   ├── prompt_loader.py
│   └── prompt_builder.py
├── README.md
├── requirements.txt
└── run.sh
```

## Data Format

Load the `dev-*.jsonl` shards in each dataset directory in filename order; each shard contains at most 100 records.

All data is stored in `jsonl` format, where each entry contains the following fields:

```json
{
    "id": "",
    "question": "",
    "data": {
        "table": {
            "header": [],
            "data": [[]]
        },
        "text": []
    },
    "answer": ""
}
```

### Field Descriptions

- **id**: A unique identifier for the question.
- **question**: The question to be answered using the provided table and text.
- **data.table.header**: The column names of the table.
- **data.table.data**: The table rows, with each row represented as a list of cell values.
- **data.text**: A list of text passages.
- **answer**: The reference answer stored as a string for evaluation.

## Benchmark Dataset

Our dataset is publicly available in our repository.

1. **[HybridQA](https://github.com/wenhuchen/HybridQA)** is a multi-hop question answering dataset built from Wikipedia tables and associated text passages, requiring reasoning across structured and unstructured information. We remove the explicit links between table cells and text passages while retaining the table contents and associated text.
2. **[OTT-QA](https://github.com/wenhuchen/OTT-QA)** is an open-domain question answering dataset over tables and text, built on HybridQA. We use the additional development questions annotated on newly collected tables beyond the original HybridQA collection, providing each question with its corresponding table and text passages.
3. **[MultiModalQA](https://github.com/allenai/multimodalqa)** is a compositional question answering dataset built from Wikipedia tables, text passages, and images. We select questions whose answers require combining table and text evidence.

## Baseline Method

| Method | Reference |
| --- | --- |
| TableLLM | [TableLLM](https://github.com/RUCKBReasoning/TableLLM) |
| TAT-LLM | [TAT-LLM](https://github.com/fengbinzhu/TAT-LLM) |
| Binder | [Binder](https://github.com/xlang-ai/Binder) |
| BlendSQL | [BlendSQL](https://github.com/parkervg/blendsql) |
| LOTUS | [LOTUS](https://github.com/lotus-data/lotus) |
| DocETL | [DocETL](https://github.com/ucbepic/docetl) |
| TableRAG | [TableRAG](https://github.com/yxh-y/TableRAG) |
| ATR | [ATR](https://github.com/ayoung206/ATR) |

## Requirements

Python 3.10 or newer is required. Python dependencies are listed in `requirements.txt`:

- `torch == 2.11.0`
- `numpy == 2.2.5`
- `transformers == 5.5.1`
- `safetensors == 0.7.0`

## Running the Code

Set the dataset and model paths in `run.sh`, then run:

```bash
bash run.sh
```

## Prompt

Method prompts are stored in `prompt/`:

- **`router.md`, `router_fused.md`**: Dynamic task planning and retrieval decisions.
- **`integrate.md`, `integrate_fused.md`**: Subquestion reasoning, evidence alignment, and relevance assessment.
- **`answer.md`**: Final answer composition.
- **`verify.md`**: Reasoning-path verification.
- **`repair_suffix.md`**: Local repair instructions.
- **`row_template.md`**: Row-description template generation.
- **`component_*.md`**: Shared stage instructions.


## Setting API Keys

Before running the code, you need to set your `model`, `api_key`, and `base_url` in `utils/llm/config.py`.

## Model Download

Due to their large size, the embedding and reranker checkpoints are hosted on Google Drive. Download them using the links below:

- **Embedding model**: [Download](https://drive.google.com/file/d/1Dmbz46sp_RG3lCvZUO4aVzzuXh3HaE_E/view?usp=sharing)
- **Reranker model**: [Download](https://drive.google.com/file/d/1FzPFh9h_dVjQpeAkVnCXxdZB44wLDw5b/view?usp=sharing)

Place the downloaded checkpoints in `model/embedding/` and `model/reranker/`, respectively.
