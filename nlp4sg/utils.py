import os
import nltk
nltk.download('punkt_tab')

import yaml
import json
import torch
import pickle
import random
import evaluate
import numpy as np

rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

def load_pkl(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_pkl(filename: str, data):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_api_key(filename):
    with open(filename, 'r') as f:
        return f.read().strip()
    
def load_config(filename):
    with open(filename, 'r') as f:
        return yaml.safe_load(f)

def load_prompt(filename):
    with open(filename, 'r') as f:
        return f.read().strip()
    
def load_type2idx(filename):
    try:
        with open(filename) as f:
            return json.load(f)
    except:
        # some datasets don't have a type
        return None
    
def load_json_schema(filename):
    try:
        with open(filename) as f:
            return json.load(f)
    except:
        # some datasets don't have a schema
        return None

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def _compute_metrics(prediction, reference):
    # rouge is case-senstive which might distort the results
    # therefore we set all words in the texts to lowercase
    rouge_results = rouge.compute(
        predictions=[prediction.lower()],
        references=[reference.lower()]
    )

    bertscore_results = bertscore.compute(
        predictions=[prediction],
        references=[reference],
        model_type="distilbert-base-uncased"
    )
    del bertscore_results["hashcode"]
    bertscore_results = {k: np.array(v) for k, v in bertscore_results.items()}

    result = {**rouge_results, **bertscore_results}
    result = {k: round(v.item() * 100, 2) for k, v in result.items()}
    return result

def compute_metrics(prediction, reference):
    if not isinstance(reference, dict):
        # rougeLSum expects newline after each sentence
        return _compute_metrics(
            prediction="\n".join(nltk.sent_tokenize(prediction.strip())), 
            reference="\n".join(nltk.sent_tokenize(reference.strip())),
        )

    results = {}
    for key in reference.keys():
        # rougeLSum expects newline after each sentence
        results[key] = _compute_metrics(
            prediction="\n".join(prediction[key].strip().split(";")), 
            reference="\n".join(reference[key].strip().split(";"))
        )

    results["overall"] = {
        "rouge1": round(np.mean([v["rouge1"] for v in results.values()]).item(), 2),
        "rouge2": round(np.mean([v["rouge2"] for v in results.values()]).item(), 2),
        "rougeL": round(np.mean([v["rougeL"] for v in results.values()]).item(), 2),
        "rougeLsum": round(np.mean([v["rougeLsum"] for v in results.values()]).item(), 2),
        "precision": round(np.mean([v["precision"] for v in results.values()]).item(), 2),
        "recall": round(np.mean([v["recall"] for v in results.values()]).item(), 2),
        "f1": round(np.mean([v["f1"] for v in results.values()]).item(), 2),
    }

    return results
