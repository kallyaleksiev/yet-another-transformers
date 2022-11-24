import time
import logging

import torch
from tqdm import tqdm


logger = logging.getLogger(__name__)


def train(model,
          *,
          optimizer,
          scheduler,
          criterion,
          train_dataloader,
          val_dataloader=None,
          num_epochs=300,
          device=torch.device("cpu"),
          cuda_rank=None
          ):
    # in a distributed scenario use the index of
    # the relevant gpu for .to(...) calls rather
    # than the torch device
    device = cuda_rank if cuda_rank is not None else device
    for epoch in range(num_epochs):
        start_epoch = time.time()
        total_loss = 0.0

        model.train()

        for batch in tqdm(train_dataloader):
            batch_ids, batch_padding_mask = batch["input_ids"].to(
                device), batch["attention_mask"].to(device)
            batch_labels = batch["label"].to(device)

            output = model(batch_ids, batch_padding_mask)
            loss = criterion(output, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        end_epoch = time.time()
        total_seconds = end_epoch - start_epoch
        log_message = f"Epoch {epoch+1}, Total Time: {total_seconds:.2f} seconds, Total Loss: {total_loss}"
        logger.info(log_message)

        torch.cuda.empty_cache()

        if val_dataloader is not None:
            accuracy, val_total_seconds = test(
                model, device=device, test_dataloader=val_dataloader)

            log_message = f"Validation completed, Accuracy: {accuracy}, Total Time: {val_total_seconds:.2f}"
            logger.info(log_message)
        else:
            logger.debug("Validation dataloader not provided, skipping")


def test(model,
         *,
         test_dataloader,
         device=torch.device("cpu"),
         cuda_rank=None
         ):
    # in a distributed scenario use the index of
    # the relevant gpu for .to(...) calls rather
    # than the torch device
    device = cuda_rank if cuda_rank is not None else device
    model.eval()
    correct = 0
    start_test = time.time()
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch_ids, batch_padding_mask = batch["input_ids"].to(
                device), batch["attention_mask"].to(device)
            batch_labels = batch["label"].to(device)

            output = model(batch_ids, batch_padding_mask)
            predictions = torch.argmax(output, dim=-1)
            correct += (predictions == batch_labels).sum().item()

    accuracy = correct / len(test_dataloader.dataset)

    end_test = time.time()
    total_seconds = end_test-start_test

    torch.cuda.empty_cache()
    return accuracy, total_seconds
