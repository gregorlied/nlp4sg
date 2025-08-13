import os
import xgrammar as xgr

from nlp4sg.utils import load_json_schema


def create_logits_processor(config, tokenizer, batch_size=1):
    
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer,
        vocab_size=len(tokenizer),
    )

    grammar_compiler = xgr.GrammarCompiler(tokenizer_info)

    json_schema_path = os.path.join(config["data"]["data_dir"], "schema.json")
    json_schema = load_json_schema(json_schema_path)

    if json_schema is None:
        return None

    compiled_grammar = grammar_compiler.compile_json_schema(json_schema)
    xgr_logits_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)
    xgr_logits_processor.batch_size = batch_size

    return [xgr_logits_processor]