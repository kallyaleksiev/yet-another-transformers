import copy

import torch.nn as nn

from transformers import AutoTokenizer


def clone(module, times):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(times)])


def tokenize(batch, passed_tokenizer):
    return passed_tokenizer(batch["text"], padding="max_length", truncation=True, return_tensors="pt")


def train_tokenizer(from_base, text_dataset, vocab_size=10000):
    base_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=from_base,
                                                   cache_dir="tokenizers")

    def text_iterator():
        step = 1000
        for i in range(0, len(text_dataset), step):
            yield text_dataset[i: i+step]["text"]
    
    return base_tokenizer.train_new_from_iterator(text_iterator(), vocab_size=vocab_size)
    