import pyarrow.parquet as pq
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import List, Dict
from torch.utils.data import DataLoader, IterableDataset
import torch


class ParquetDataset(Dataset):
    def __init__(
        self,
        parquet_file: str,
        tokenizer: str,
        sequence_length: int,
        training_samples: int,
    ):
        self.parquet_ds = pq.read_table(parquet_file, memory_map=True)
        self.real_length = len(self.parquet_ds)
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.training_samples = training_samples

    def __len__(self):
        return self.training_samples

    def __getitem__(self, idx: int):
        sample_str = str(self.parquet_ds["text"][idx % self.real_length])
        return self.tokenizer.encode_plus(
            sample_str,
            max_length=self.sequence_length + 1,
            padding="max_length",
            truncation=True,
            padding_side="right",
        )


@dataclass
class CollatorForCLM:
    sequence_length: int
    pad_token_id: int

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.LongTensor(
            [examples[i]["input_ids"] for i in range(len(examples))]
        )  # (b, s+1)
        inputs = input_ids[:, :-1].clone()
        labels = input_ids[:, 1:]
        # For padding tokens, mask the loss
        labels[labels == self.pad_token_id] = -100
        assert inputs.shape[1] == labels.shape[1] == self.sequence_length
        assert inputs.shape == labels.shape
        return inputs, labels


class IterableParquetDataset(IterableDataset):
    def __init__(
        self, parquet_file: str, tokenizer, sequence_length: int, bos_token_id: int = 1
    ):
        self.parquet_ds = pq.read_table(parquet_file, memory_map=True)
        self.real_length = len(self.parquet_ds)
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.bos_token_id = bos_token_id
        self.current_index = 0
        self.token_buffer = []

    def __iter__(self):
        # Reset buffer and index when starting a new iteration
        self.token_buffer = []
        self.current_index = 0
        return self

    def __next__(self):
        # Keep filling a buffer until we have enough tokens for a new sample.
        # Mask the loss for each token following the BoS token using -100 index.
        # Add your implementation here
        self.token_buffer = []

        while len(self.token_buffer) < self.sequence_length + 1:
            sample_str = str(
                self.parquet_ds["text"][self.current_index % self.real_length]
            )
            tokens = self.tokenizer.encode_plus(
                sample_str,
                padding=False,
                truncation=True,
                max_length=self.sequence_length + 1,
            )
            self.token_buffer.extend(tokens["input_ids"])
            self.current_index += 1

        self.current_index -= 1
        self.token_buffer = self.token_buffer[: self.sequence_length + 1]
        # Get the next sample from the buffer
        inputs = torch.LongTensor(self.token_buffer[:-1])
        labels = torch.LongTensor(self.token_buffer[1:])
        # Mask the loss for padding tokens
        labels[inputs == self.bos_token_id] = -100
        labels[labels == self.bos_token_id] = -100
        return inputs, labels


if __name__ == "__main__":
    # Create dataset instance
    dataset_path = "/capstor/store/cscs/ethz/large-sc/datasets/train_data.parquet"
    sequence_length = 4096
    batch_size = 32

    tokenizer = AutoTokenizer.from_pretrained("unsloth/Mistral-Nemo-Base-2407-bnb-4bit")
    # Create dataset (only requesting 1 sample)
    dataset = ParquetDataset(
        parquet_file=dataset_path,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        training_samples=32,
    )

    # Get the first sample
    sample = dataset[0]
    sample = sample["input_ids"][:200]
    dec_sample = tokenizer.decode(sample)
    print(f"Decoded sample: {dec_sample}")

    # Create collator
    collator = CollatorForCLM(
        sequence_length=sequence_length, pad_token_id=tokenizer.pad_token_id
    )
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

    # Get a batch using a for loop
    for batch_inputs, batch_labels in dataloader:
        # Print shapes
        print(f"Input shape: {batch_inputs.shape}")
        print(f"Labels shape: {batch_labels.shape}")
        # Count ignored tokens in the loss calculation
        ignored_count = (batch_labels == -100).sum().item()
        total_label_tokens = batch_labels.numel()
        print(
            f"Ignored tokens in loss: {ignored_count} out of {total_label_tokens} ({ignored_count/total_label_tokens*100:.2f}%)"
        )
        # Only process the first batch
        break

    # Create iterable dataset
    iterable_dataset = IterableParquetDataset(
        parquet_file=dataset_path,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
    )
    # Create iterable dataloader
    iterable_dataloader = DataLoader(iterable_dataset, batch_size=batch_size)
    # Get a batch using a for loop
    for batch_inputs, batch_labels in iterable_dataloader:
        # Print shapes
        print(f"Input shape: {batch_inputs.shape}")
        print(f"Labels shape: {batch_labels.shape}")
        # Count ignored tokens in the loss calculation
        ignored_count = (batch_labels == -100).sum().item()
        total_label_tokens = batch_labels.numel()
        print(
            f"Ignored tokens in loss: {ignored_count} out of {total_label_tokens} ({ignored_count/total_label_tokens*100:.2f}%)"
        )
        # Only process the first batch
        break
