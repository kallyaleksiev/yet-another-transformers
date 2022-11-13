import argparse
import os
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

from comp_utils import tokenize
from models import Test_RottenTomatoes_Classifier
from procedures import train, test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="rhea", description="Driver program for experimentation")

    parser.add_argument("--device",
                        type=str,
                        choices=["cpu", "cuda", "mps"],
                        help="the kind of device to use - one of cpu, gpu, or mps")

    parser.add_argument("--epochs",
                        type=int,
                        default=300,
                        help="the number of training epochs to run")

    parser.add_argument("--num-workers",
                        type=int,
                        default=4,
                        help="the number of workers for the dataloaders")

    parser.add_argument("--batch-size",
                        type=int,
                        default=16,
                        help="the batch size for all datasets and splits")

    parser.add_argument("--dataset",
                        type=str,
                        default="rotten_tomatoes",
                        choices=["rotten_tomatoes"],
                        help="the dataset to use, currently only rotten_tomatoes is supported"
                        )

    parser.add_argument("--base-tokenizer",
                        type=str,
                        default="bert-base-uncased",
                        help="the tokenizer to use"
                        )

    args = parser.parse_args()

    os.makedirs("datasets", exist_ok=True)
    os.makedirs("tokenizers", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.base_tokenizer,
                                              cache_dir="tokenizers")

    VOCAB_SIZE = tokenizer.vocab_size

    train_dataset = load_dataset(path=args.dataset,
                                 cache_dir="datasets",
                                 split="train")
    val_dataset = load_dataset(path=args.dataset,
                               cache_dir="datasets",
                               split="validation")
    # test_dataset = load_dataset(path=args.dataset,
    #                             cache_dir="datasets",
    #                             split="test")

    tokenization = partial(tokenize, passed_tokenizer=tokenizer)

    tokenized_train = train_dataset.map(tokenization, batched=True)
    tokenized_val = val_dataset.map(tokenization, batched=True)
    # tokenized_test = test_dataset.map(tokenization, batched=True)

    tokenized_train.set_format(type="torch", columns=[
        "input_ids", "attention_mask", "label"])
    tokenized_val.set_format(type="torch", columns=[
        "input_ids", "attention_mask", "label"])
    # tokenized_test.set_format(type="torch", columns=[
    #                           "input_ids", "attention_mask", "label"])

    train_dataloader = DataLoader(tokenized_train,
                                  num_workers=args.num_workers,
                                  batch_size=args.batch_size,
                                  shuffle=True)

    val_dataloader = DataLoader(tokenized_val,
                                num_workers=args.num_workers,
                                batch_size=args.batch_size,
                                shuffle=True)

    # test_dataloader = DataLoader(tokenized_test,
    #                              num_workers=args.num_workers,
    #                              batch_size=args.batch_size,
    #                              shuffle=True)

    model = Test_RottenTomatoes_Classifier(vocab_size=VOCAB_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-5)
    device = torch.device(args.device)

    train(model,
          optimizer=optimizer,
          criterion=criterion,
          train_dataloader=train_dataloader,
          device=device,
          val_dataloader=val_dataloader,
          num_epochs=args.epochs)
