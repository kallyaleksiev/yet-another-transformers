import argparse
import os
from functools import partial
import logging

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.optim as optim
import torch.utils.data as data_utils

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

from comp_utils import tokenize, train_tokenizer
from comp_utils import tokenize
from models import Test_RottenTomatoes_Classifier
from procedures import train, test
from ddp_utils import setup, cleanup


logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%m/%d/%Y %I:%M:%S %p",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def worker(rank, world_size, cli_args, partitions, val_dataloader, vocab_size):
    r"""The computation that will be performed by a single GPU
    """
    setup(rank, world_size, backend="gloo")
    model = DDP(Test_RottenTomatoes_Classifier(
        vocab_size=vocab_size).to(rank), device_ids=[rank], find_unused_parameters=True)

    train_dataloader = DataLoader(partitions[rank],
                                  num_workers=cli_args.num_workers,
                                  batch_size=cli_args.batch_size,
                                  shuffle=True)

    steps_per_epoch = len(train_dataloader)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=5e-3,
                           betas=(0.9, 0.98),
                           weight_decay=1e-5)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                              max_lr=0.1,
                                              epochs=cli_args.epochs,
                                              steps_per_epoch=steps_per_epoch,
                                              anneal_strategy="linear",
                                              div_factor=20)

    device = torch.device(cli_args.device)

    if rank == 0:
        train(model,
              optimizer=optimizer,
              scheduler=scheduler,
              criterion=criterion,
              train_dataloader=train_dataloader,
              val_dataloader=val_dataloader,
              num_epochs=cli_args.epochs,
              cuda_rank=rank)

        # at the end of training, do one final test in the master process
        test_dataset = load_dataset(path=cli_args.dataset,
                                    cache_dir="datasets",
                                    split="test")
        tokenized_test = test_dataset.map(tokenization, batched=True)
        tokenized_test.set_format(type="torch", columns=[
            "input_ids", "attention_mask", "label"])
        test_dataloader = DataLoader(tokenized_test,
                                     num_workers=cli_args.num_workers,
                                     batch_size=cli_args.batch_size,
                                     shuffle=True)

        test(model=model, test_dataloader=test_dataloader, cuda_rank=rank)
    else:
        train(model,
              optimizer=optimizer,
              scheduler=scheduler,
              criterion=criterion,
              train_dataloader=train_dataloader,
              val_dataloader=None,
              num_epochs=cli_args.epochs,
              cuda_rank=rank)


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

    parser.add_argument("--base-tokenizer",
                        type=str,
                        default="bert-base-uncased",
                        help="the tokenizer to use"
                        )

    parser.add_argument("--train-new-tokenizer",
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help="whether to train a new tokenizer for rotten tomatoes"
                        )

    parser.add_argument("--vocab-size",
                        type=int,
                        default=10000,
                        help="if training a new tokenizer, what vocabulary size to use"
                        )

    parser.add_argument("--dataset",
                        type=str,
                        default="rotten_tomatoes",
                        choices=["rotten_tomatoes"],
                        help="the dataset to use, currently only rotten_tomatoes is supported"
                        )

    args = parser.parse_args()

    os.makedirs("datasets", exist_ok=True)
    os.makedirs("tokenizers", exist_ok=True)

    train_dataset = load_dataset(path=args.dataset,
                                 cache_dir="datasets",
                                 split="train")
    val_dataset = load_dataset(path=args.dataset,
                               cache_dir="datasets",
                               split="validation")

    if not args.train_new_tokenizer:
        logger.info(
            f"Using pre-trained base tokenizer for: {args.base_tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.base_tokenizer,
                                                  cache_dir="tokenizers")
    else:
        logger.info(
            f"Training new tokenizer from base {args.base_tokenizer} with vocab_size: {args.vocab_size}")
        text_dataset = load_dataset(path=args.dataset,
                                    cache_dir="datasets",
                                    split="all")
        tokenizer = train_tokenizer(
            from_base=args.base_tokenizer, text_dataset=text_dataset, vocab_size=args.vocab_size)

    VOCAB_SIZE = tokenizer.vocab_size

    tokenization = partial(tokenize, passed_tokenizer=tokenizer)

    tokenized_train = train_dataset.map(tokenization, batched=True)
    tokenized_val = val_dataset.map(tokenization, batched=True)

    tokenized_train.set_format(type="torch", columns=[
        "input_ids", "attention_mask", "label"])
    tokenized_val.set_format(type="torch", columns=[
        "input_ids", "attention_mask", "label"])

    val_dataloader = DataLoader(tokenized_val,
                                num_workers=args.num_workers,
                                batch_size=args.batch_size,
                                shuffle=True)

    world_size = args.world_size
    lengths = [len(train_dataset) // world_size for _ in range(world_size)]
    lengths[-1] += len(train_dataset) - sum(lengths)

    partitions = data_utils.random_split(tokenized_train, lengths=lengths)

    mp.spawn(
        worker,
        args=(world_size, args, partitions, val_dataloader, VOCAB_SIZE),
        nprocs=world_size,
    )

    cleanup()
