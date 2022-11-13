import copy

import torch.nn as nn


def clone(module, times):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(times)])


def tokenize(batch, passed_tokenizer):
    return passed_tokenizer(batch["text"], padding="max_length", truncation=True, return_tensors="pt")
