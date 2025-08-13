import os
import pandas as pd
from datasets import Dataset

from nlp4sg.utils import load_prompt, load_type2idx


def tokenize(example, tokenizer, max_length, max_target_length, add_prefix_token=False):
    text = example["text"]
    summary = example["summary"]

    if add_prefix_token:
        text = "summarize: " + text

    model_inputs = tokenizer(
        text,
        padding=False,
        truncation=True,
        max_length=max_length,
    )

    model_inputs["labels"] = tokenizer(
        summary,
        padding=False,
        truncation=True,
        max_length=max_target_length,
    ).input_ids

    if "type" in example: 
        model_inputs["type"] = example["type"]

    return model_inputs


def create_messages(example, prompt):
    return {
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": example["text"]},
            {"role": "assistant", "content": example["summary"]}
        ]
    }


def create_prompt_completion(example, prompt):
    return {
        "prompt": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": example['text']},
        ], 
        "completion": [
            {"role": "assistant", "content": example['summary']},
        ]
    }


def create_test_messages(example, tokenizer, config, prompt):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": example["text"]},
    ]

    example["text"] = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False, # only relevant for qwen
    )

    model_inputs = tokenize(
        example, 
        tokenizer, 
        max_length=config["data"]["max_length"], 
        max_target_length=config["data"]["max_target_length"], 
    )

    return model_inputs


def get_train_map_fn_and_kwargs(config, tokenizer):
    if config["model"]["is_encoder_decoder"]:
        map_fn = tokenize
        map_fn_kwargs = {
            "tokenizer": tokenizer,
            "max_length": config["data"]["max_length"],
            "max_target_length": config["data"]["max_target_length"],
            "add_prefix_token": True,
        }
        return map_fn, map_fn_kwargs
    
    map_fn = create_messages if config["model"]["use_chat_template"] else create_prompt_completion
    prompt_path = os.path.join(config["data"]["data_dir"], "prompt.txt")
    map_fn_kwargs = {
        "prompt": load_prompt(prompt_path),
    }
    return map_fn, map_fn_kwargs


def get_test_map_fn_and_kwargs(config, tokenizer):
    if config["model"]["is_encoder_decoder"]:
        map_fn = tokenize
        map_fn_kwargs = {
            "tokenizer": tokenizer,
            "max_length": config["data"]["max_length"],
            "max_target_length": config["data"]["max_target_length"],
            "add_prefix_token": True,
        }
        return map_fn, map_fn_kwargs
    
    map_fn = create_test_messages 
    prompt_path = os.path.join(config["data"]["data_dir"], "prompt.txt")
    map_fn_kwargs = {
        "prompt": load_prompt(prompt_path),
        "tokenizer": tokenizer,
        "config": config,
    }
    return map_fn, map_fn_kwargs


def create_train_dataset(config, tokenizer):
    # load dataset
    data_path = os.path.join(config["data"]["data_dir"], "train.csv")
    train_df = pd.read_csv(data_path)
    train_ds = Dataset.from_pandas(train_df)
    if config["debug"]: train_ds = train_ds.select(range(100))

    # apply type2idx mapping, if available
    type2idx_path = os.path.join(config["data"]["data_dir"], "type2idx.json")
    type2idx = load_type2idx(type2idx_path)
    if type2idx is not None:
        train_ds = train_ds.map(lambda example: {"type": type2idx[example["type"]]})

    # prepare data
    map_fn, map_fn_kwargs = get_train_map_fn_and_kwargs(config, tokenizer)
    train_ds = train_ds.map(map_fn, fn_kwargs=map_fn_kwargs, remove_columns=train_ds.features, batched=False)
    return train_ds


def create_test_dataset(config, tokenizer):
    # load dataset
    data_path = os.path.join(config["data"]["data_dir"], "test.csv")
    test_df = pd.read_csv(data_path)
    test_ds = Dataset.from_pandas(test_df)
    if config["debug"]: test_ds = test_ds.select(range(20))

    # apply type2idx mapping, if available
    type2idx_path = os.path.join(config["data"]["data_dir"], "type2idx.json")
    type2idx = load_type2idx(type2idx_path)
    if type2idx is not None:
        test_ds = test_ds.map(lambda example: {"type": type2idx[example["type"]]})

    # prepare data
    map_fn, map_fn_kwargs = get_test_map_fn_and_kwargs(config, tokenizer)
    test_ds = test_ds.map(map_fn, fn_kwargs=map_fn_kwargs, remove_columns=test_ds.features, batched=False)
    return test_ds