
from transformers import TrainingArguments, Trainer
from trl import SFTConfig, SFTTrainer


def create_trainer(config, model, tokenizer, train_ds):

    if config["debug"]:
        print("Debug. Set num_train_epochs = 1.")
        config["training"]["num_train_epochs"] = 1

    if model.config.is_encoder_decoder:

        args = TrainingArguments(
            output_dir=config["output_dir"],
            optim=config["training"]["optim"],
            num_train_epochs=config["training"]["num_train_epochs"],
            learning_rate=config["training"]["learning_rate"],
            lr_scheduler_type=config["training"]["lr_scheduler_type"],
            warmup_ratio=config["training"]["warmup_ratio"],
            per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
            gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
            gradient_checkpointing=config["training"]["gradient_checkpointing"],
            max_grad_norm=config["training"]["max_grad_norm"],
            bf16=config["training"]["bf16"],
            tf32=config["training"]["tf32"],
            report_to="wandb",
            logging_steps=10,
            save_strategy="epoch",
            label_names=["labels"],
        )

        trainer = Trainer(
            args=args,
            model=model,
            train_dataset=train_ds,
            processing_class=tokenizer,
        )

        return trainer

    max_seq_length = config["data"]["max_length"] + config["data"]["max_target_length"]

    args = SFTConfig(
        output_dir=config["output_dir"],
        optim=config["training"]["optim"],
        num_train_epochs=config["training"]["num_train_epochs"],
        learning_rate=config["training"]["learning_rate"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        warmup_ratio=config["training"]["warmup_ratio"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        max_grad_norm=config["training"]["max_grad_norm"],
        bf16=config["training"]["bf16"],
        tf32=config["training"]["tf32"],
        report_to="wandb",
        logging_steps=10,
        save_strategy="epoch",
        label_names=["labels"],
        packing=False,
        max_seq_length=max_seq_length,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False, # No need to add additional separator token
        }
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    return trainer