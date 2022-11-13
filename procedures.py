import time
import logging

import torch
from tqdm import tqdm

logger = logging.getLogger("rhea")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s %(name)s [%(levelname)s] %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")
handler.setFormatter(formatter)


def train(model,
          *,
          optimizer,
          criterion,
          train_dataloader,
          device,
          val_dataloader=None,
          num_epochs=300,
          ):
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

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        end_epoch = time.time()
        total_seconds = end_epoch - start_epoch
        log_message = f"Epoch {epoch+1}, Total Time: {total_seconds} seconds, Total Loss: {total_loss}"
        logging.info(log_message)

        if val_dataloader is not None:
            accuracy, val_total_seconds = test(
                model, device=device, test_dataloader=val_dataloader)

            log_message = f"Validation completed, Accuracy: {accuracy}, Total Time: {val_total_seconds}"
            logging.info(log_message)
        else:
            logging.debug("Validation dataloader not provided, skipping")


def test(model,
         *,
         device,
         test_dataloader,
         ):
    model.eval()
    correct = 0
    start_test = time.time()
    with torch.no_grad():
        for batch in test_dataloader:
            batch_ids, batch_padding_mask = batch["input_ids"].to(
                device), batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            output = model(batch_ids, batch_padding_mask)
            predictions = torch.argmax(output, dim=-1)
            correct += (predictions == batch_labels).sum().item()

    accuracy = correct / len(test_dataloader.dataset)

    end_test = time.time()
    total_seconds = end_test-start_test

    return accuracy, total_seconds
