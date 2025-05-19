import os
import signal
import sys
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import CollatorForCLM, ParquetDataset
from model import Transformer, TransformerModelArgs
from utils import build_lr_scheduler, clip_grad_norm_, get_args, init_logger, logger, PRECISION_STR_TO_DTYPE, set_default_dtype, catch_SIG_exception, handle_exit

def train(args):
  
  logger.info(f"Experiment args: {args}")
  # Init
  device = torch.device(f"cuda:{(os.getenv('LOCAL_RANK', 0))}")
  model_dtype = PRECISION_STR_TO_DTYPE[args.model_dtype]

  # load checkpoint if provided
  if args.checkpoint_id:
    logger.info(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(os.path.join(args.checkpoint_path, f"checkpoint_{args.checkpoint_id}.ckpt"), map_location="cpu")
  else:
     checkpoint = None

  # Set up DataLoader
  logger.info("Setting up DataLoaders...")
  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
  train_ds = ParquetDataset(args.dataset, tokenizer, args.sequence_length, args.batch_size*args.training_steps)
  train_collator = CollatorForCLM(args.sequence_length, tokenizer.pad_token_id)
  train_dl = DataLoader(train_ds,
                        batch_size=args.batch_size,
                        collate_fn=train_collator)
  train_dl_iterator = iter(train_dl)

  if checkpoint is not None:
    # Skip the already processed batches in the previous run
    for _ in range(checkpoint["training_step"]):
      next(train_dl_iterator)

  # Set up Model
  logger.info("Setting up Model...")
  model_config = TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        vocab_size=tokenizer.vocab_size,
        seq_len=args.sequence_length,
    )
  with set_default_dtype(model_dtype):
    model = Transformer(model_config)
    if checkpoint is not None:
      model.load_state_dict(checkpoint["model"], strict=False)
      logger.info("Model loaded from checkpoint")
  model.to(device)

  if args.compile:
    logger.info("Using `torch.compile`")
    model = torch.compile(model, fullgraph=True)
  
  model.train()

  # Build Optimizers & LR Scheduler
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, fused=args.fused_optimizer)
  # Load optimizer state if checkpoint is provided
  if checkpoint is not None:
    optimizer.load_state_dict(checkpoint["optimizer"])
    logger.info("Optimizer loaded from checkpoint")
  # Load LR Scheduler state if checkpoint is provided
  lr_scheduler = build_lr_scheduler(optimizer, args.lr_warmup_steps)
  if checkpoint is not None:
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    logger.info("LR Scheduler loaded from checkpoint")
  # Load current training_step if checkpoint is provided
  if checkpoint is not None:
    training_step = checkpoint["training_step"]
    logger.info(f"Resuming training from training_step {training_step}")
  else:
    training_step = 0
    logger.info("Starting training!")

  del checkpoint

  # register SIGTERM handler, binding in your objects
  signal.signal(signal.SIGUSR1, catch_SIG_exception)
  signal.signal(signal.SIGTERM, catch_SIG_exception)
  try:
    while training_step < args.training_steps:
        input_ids, labels = next(train_dl_iterator)
        num_items_in_batch = labels.ne(-100).sum()
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(0, 1), reduction="sum")
        loss = loss / num_items_in_batch
        del logits
        loss.backward()

        # Clip gradients
        clip_grad_norm_(model.parameters(), args.grad_max_norm)

        optimizer.step()
        lr_scheduler.step()
        # If specified in the args, raise dummy error to test the handler
        if args.raise_error and (training_step == args.error_step):
           raise Exception("Simulated exception to test signal handler", -1)
        # Logging
        if (training_step == 1 or training_step % args.logging_frequency == 0):
          logger.info(f"Training step: {training_step} | Loss: {loss.item():.2f}")
        training_step += 1
    logger.info("Training completed")
    sys.exit(0)

  except Exception as e:
        # if error has no signal, set it to -1, since it is the error coming from the code
        if len(e.args) < 2:
            error_type = -1
        else:
            error_type = e.args[1]
        # on any exception, invoke the same handler
        handle_exit(model, optimizer, lr_scheduler, training_step, args.checkpoint_path, error_type, logger)
        sys.exit(0)

if __name__ == "__main__":
  init_logger()
  args = get_args()
  train(args)