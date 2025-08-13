import torch

class NLP4SGCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        output["labels"] = [sample["labels"] for sample in batch]
        if "type" in batch[0]: output["type"] = [sample["type"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(input_ids) for input_ids in output["input_ids"]])

        # while using fp16 we will pad the sequence to a multiple of `pad_to_multiple_of`
        if self.pad_to_multiple_of:
            batch_max = batch_max + (self.pad_to_multiple_of - (batch_max % self.pad_to_multiple_of))

        # add padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["input_ids"]]
            output["attention_mask"] = [s + (batch_max - len(s)) * [0] for s in output["attention_mask"]]
            output["labels"] = [s + [-100] * (batch_max - len(s)) for s in output["labels"]]
        else:
            output["input_ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["input_ids"]]
            output["attention_mask"] = [(batch_max - len(s)) * [0] + s for s in output["attention_mask"]]
            output["labels"] = [(batch_max - len(s)) * [-100] + s for s in output["labels"]]

        # convert to tensors
        output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)
        output["labels"] = torch.tensor(output["labels"], dtype=torch.long)
        if "type" in batch[0]: output["type"] = torch.tensor(output["type"], dtype=torch.long)

        return output
