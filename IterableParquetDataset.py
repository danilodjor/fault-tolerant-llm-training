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
