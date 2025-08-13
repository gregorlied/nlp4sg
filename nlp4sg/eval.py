import argparse

import os
import ast
import torch
import huggingface_hub
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from nlp4sg.collator import NLP4SGCollator
from nlp4sg.dataset import create_test_dataset
from nlp4sg.model import create_model_and_tokenizer
from nlp4sg.processor import create_logits_processor
from nlp4sg.utils import load_api_key, load_config, load_type2idx, seed_everything, compute_metrics

def postprocess(prediction, reference):
    reference = ast.literal_eval(reference)
    try:
        prediction = ast.literal_eval(prediction)
    except:
        prediction = {
            'life_style': '',
            'family_history': '',
            'social_history': '',
            'medical_surgical_history': '',
            'signs_symptoms': '',
            'comorbidities': '',
            'diagnostic_techniques_procedures': '',
            'diagnosis': '',
            'laboratory_values': '',
            'pathology': '',
            'pharmacological_therapy': '',
            'interventional_therapy': '',
            'patient_outcome_assessment': '',
            'age': '',
            'gender': '',
        }
    return prediction, reference


def eval(config):

    # setup
    print("Setup evaluation...")
    seed_everything(config["training"]["seed"])

    hf_token = load_api_key('./secrets/hf_token.txt')
    huggingface_hub.login(token=hf_token)
    print("Done.\n")

    # create model and tokenzier
    print("Load model and tokenizer from checkpoint...")
    model_name = config["model"]["model_name"]
    config["model"]["model_name"] = config["output_dir"]
    model, tokenizer = create_model_and_tokenizer(config)  
    config["model"]["is_encoder_decoder"] = model.config.is_encoder_decoder
    model.eval()
    print("Done.\n")

    # this appears to be needed for fine-tuned Qwen models 
    if model_name in ["Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-0.6B"]:
        model = model.to(dtype=torch.float16)
    
    # create dataset
    print("Create dataset...")
    test_ds = create_test_dataset(config, tokenizer)
    print("Done.\n")

    # create dataloader
    print("Create dataloader...")
    batch_size = 1
    val_collate_fn = NLP4SGCollator(tokenizer)
    test_dataloader = DataLoader(test_ds, shuffle=False, batch_size=batch_size, collate_fn=val_collate_fn)
    print("Done.\n")

    # run inference
    print("Run inference...")
    types, references, predictions = [], [], []
    for batch in tqdm(test_dataloader):
        batch = {k: v.to(device=model.device) for k, v in batch.items()}

        batch_labels = batch["labels"].tolist()

        with torch.no_grad():

            if model.config.is_encoder_decoder:
                batch_predictions = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **config["generation"]
                )
                
                batch_predictions = batch_predictions.tolist()
            else:
                try:
                    # We cannot reset here because __call__ is not invoked when stop token is sampled.
                    # Therefore, each `generate()` call needs to instantiate an LogitsProcessor.
                    logits_processor = None
                    if config["model"]["use_structured_output_generation"]:
                        logits_processor = create_logits_processor(config, tokenizer, batch_size=batch_size)

                    batch_predictions = model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        logits_processor=logits_processor,
                        **config["generation"]
                    )

                    batch_predictions = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(batch["input_ids"], batch_predictions)
                    ]
                except:
                    print("Structured generation failed.")
                    batch_predictions = [[tokenizer.pad_token_id]]

            if config["debug"]:
                print(tokenizer.batch_decode(batch_predictions, skip_special_tokens=True))

        if "type" in batch: types.extend(batch["type"].tolist())
        references.extend([[x for x in xs if x != -100] for xs in batch_labels])
        predictions.extend([[x for x in xs if x != -100] for xs in batch_predictions])
    print("Done.\n")

    print("Compute metrics...\n")
    metrics, decoded_references, decoded_predictions = [], [], []
    for prediction, reference in tqdm(zip(predictions, references), total=len(predictions)):
        decoded_prediction = tokenizer.decode(prediction, skip_special_tokens=True)
        decoded_predictions.append(decoded_prediction)
        decoded_reference = tokenizer.decode(reference, skip_special_tokens=True)
        decoded_references.append(decoded_reference)
        if decoded_reference.startswith("{") and decoded_reference.endswith("}"):
            decoded_prediction, decoded_reference = postprocess(decoded_prediction, decoded_reference)
        metrics.append(compute_metrics(decoded_prediction, decoded_reference))
    print("Done.\n")

    print("Save metrics to file...\n")
    data = {"predictions": decoded_predictions, "reference": decoded_references}
    if len(types) > 0: 
        # in this case the path is guaranteed to exist
        type2idx_path = os.path.join(config["data"]["data_dir"], "type2idx.json")
        type2idx = load_type2idx(type2idx_path)
        idx2type = {v: k for k, v in type2idx.items()}
        data["type"] = [idx2type[x] for x in types]
    df = pd.DataFrame(data)
    df = pd.concat([df, pd.DataFrame(metrics)], axis=1)

    print(f"\t   Mean  Median")
    print(f"Rouge1: {df['rouge1'].mean():>8.2f} {df['rouge1'].median():.2f}")
    print(f"Rouge2: {df['rouge2'].mean():>8.2f} {df['rouge2'].median():.2f}")
    print(f"RougeL: {df['rougeL'].mean():>8.2f} {df['rougeL'].median():.2f}")
    print(f"RougeLsum: {df['rougeLsum'].mean():>5.2f} {df['rougeLsum'].median():.2f}")
    print(f"Precision: {df['precision'].mean():>5.2f} {df['precision'].median():.2f}")
    print(f"Recall: {df['recall'].mean():>8.2f} {df['recall'].median():.2f}")
    print(f"F1: {df['f1'].mean():>12.2f} {df['f1'].median():.2f}")

    result_path = os.path.join(config["output_dir"], "results.csv")
    df.to_csv(result_path, index=False)

    print("\nDone.\n")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type=str, 
        default= "./configs/flan_t5_base.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    eval(config)
