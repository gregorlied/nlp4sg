import argparse

import huggingface_hub
import wandb
from peft import PeftModel

from nlp4sg.dataset import create_train_dataset
from nlp4sg.model import create_model_and_tokenizer
from nlp4sg.trainer import create_trainer
from nlp4sg.utils import load_api_key, load_config, seed_everything

def train(config):

    # setup
    print("Setup training...")
    seed_everything(config["training"]["seed"])

    hf_token = load_api_key('./secrets/hf_token.txt')
    huggingface_hub.login(token=hf_token)

    wandb_key = load_api_key('./secrets/wandb_key.txt')
    wandb.login(key=wandb_key)

    wandb.init(
        project=config["wandb"]["project"], 
        entity=config["wandb"]["entity"],
    )
    print("Done.\n")

    # create model and tokenzier
    print("Create model and tokenizer...")
    model, tokenizer = create_model_and_tokenizer(config)  
    config["model"]["is_encoder_decoder"] = model.config.is_encoder_decoder
    print("Done.\n")
    
    # create dataset
    print("Create dataset...")
    train_ds = create_train_dataset(config, tokenizer)
    print("Done.\n")
    
    # create trainer
    print("Create trainer...")
    trainer = create_trainer(config, model, tokenizer, train_ds)
    print("Done.\n")

    # start training
    print("Start training...")
    trainer.train()
    print("Done.\n")

    # save final checkpoint
    print("Save final checkpoint...")
    model = trainer.model
    if isinstance(model, PeftModel):
        model = model.merge_and_unload()

    output_dir = config["output_dir"]
    model.save_pretrained(output_dir)
    model.config.save_pretrained(output_dir)
    trainer.processing_class.save_pretrained(output_dir)
    print("Done.")

    wandb.finish()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type=str, 
        default= "./configs/config.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    train(config)
