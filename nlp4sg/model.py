import torch
from transformers import (
    AutoConfig,
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from trl import setup_chat_format
from peft import TaskType, LoraConfig, get_peft_model


def create_model_and_tokenizer(config):

    model_name = config["model"]["model_name"]
    model_dtype = config["model"]["model_dtype"]
    use_lora = config["model"]["use_lora"]

    config = AutoConfig.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side='right',
        trust_remote_code=True,
    )

    bnb_config = None
    if model_dtype == "fp32":
        torch_dtype = torch.float32
    elif model_dtype == "fp16":
        torch_dtype = torch.float16
    elif model_dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif model_dtype == "int4":
        torch_dtype = torch.bfloat16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quanty_typ="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quanty=True,
        )
    else:
        raise ValueError("Unknown value {}. Please choose from ['int4', 'int8', 'bf16', 'fp16', 'fp32'].")

    if config.is_encoder_decoder:
        task_type = TaskType.SEQ_2_SEQ_LM

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch_dtype,
            quantization_config=bnb_config,
            attn_implementation="eager",
            trust_remote_code=True,
        )

    else:
        task_type = TaskType.CAUSAL_LM

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch_dtype,
            quantization_config=bnb_config,
            attn_implementation="eager",
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            print("Add <pad> token.")
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            model.resize_token_embeddings(len(tokenizer))

        if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
            print("Setup chat template.")
            model, tokenizer = setup_chat_format(model, tokenizer)

    if use_lora:
        peft_config = LoraConfig(
            task_type=task_type,
            r=256,
            lora_alpha=128,
            lora_dropout=0.05,
            bias="none",
            target_modules="all-linear",
            modules_to_save=["lm_head", "embed_token"],
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, tokenizer