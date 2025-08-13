# NLP for Social Good - Text Summarization

This repository contains training and inference code developed as part of the "NLP for Social Good" course at TU Berlin.

For quick demonstrations, please visit the following Hugging Face Spaces:

- [Medical Text Summarization](https://huggingface.co/spaces/gregorlied/medical-text-summarization)
- [News, SciTLDR, and Dialog Summarization](https://huggingface.co/spaces/gregorlied/news-scitldr-dialog-summarization)


Please find the code for each demonstration under the corresponding **Files** tab on the Hugging Face Space.

## Setup

Step 1 – Store your API keys in the following files:

```bash
secrets/hf_token.txt
secrets/wandb_key.txt
```

Step 2 – Prevent these files from being committed to GitHub:

```bash
git update-index --skip-worktree secrets/hf_token.txt
git update-index --skip-worktree secrets/wandb_key.txt
```

Step 3 – Install dependencies from `pyproject.toml`:

```bash
python -m pip install --upgrade pip
pip install -e .
```

## Data preperation

Download the datasets from the [shared drive](https://drive.google.com/drive/folders/1ZgtFEc-UYp7kCviZE9okj9fSIqZIVHCB?usp=drive_link) and place them in the following structure: 

```text
data/clinical_report_summarization/train.csv
data/clinical_report_summarization/test.csv

data/news_scitldr_dialog_summarization/train.csv
data/news_scitldr_dialog_summarization/test.csv
```

## Training

To start training, run:

```python
python nlp4sg/train.py --config configs/config.yaml
```

The final checkpoint will be saved to the directory specified in `config['output_dir']`.

## Evaluation

To evaluate a trained model, run:

```python
python3 nlp4sg/evaluate.py --config configs/config.yaml
```