import argparse
import functools
import logging
import os
from contextlib import contextmanager

import torch
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger()
WORKDIR = os.getenv("WORKDIR", "")
JOBID = os.environ.get("SLURM_JOB_ID")

PRECISION_STR_TO_DTYPE = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}

def init_logger():
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def build_lr_scheduler(optimizer: torch.optim, warmup_steps: int):

    def linear_warmup_constant(
        warmup_steps: int, current_step: int
    ) -> float:
        """Computes linear warmup followed by linear decay.

        Per LambdaLR requirement, this is accomplished by returning
        a multiplicative factor to adjust the learning rate to
        create the desired schedule.
        """
        if current_step < warmup_steps:
            # linear warmup
            # 0-indexed step, hence + 1 adjustments
            current_step += 1
            curr_adjustment = float(current_step / (warmup_steps + 1))

        else:
            # constant
            curr_adjustment = 1

        return curr_adjustment

    lr_lambda = functools.partial(linear_warmup_constant, warmup_steps)
    return LambdaLR(optimizer, lr_lambda)
    
@torch.no_grad()
def clip_grad_norm_(parameters, grad_max_norm):
  grads = [p.grad for p in parameters if p.grad is not None]
  total_norm = torch.nn.utils.get_total_norm(grads, error_if_nonfinite=True)
  torch.nn.utils.clip_grads_with_norm_(parameters, grad_max_norm, total_norm)
  return total_norm

def handle_exit(model, optimizer, lr_scheduler, training_step, checkpoint_path, type, logger):
    """Handle exit based on the type of error."""
    if type == 15:
        logger.info(f"[EXIT HANDLER] Job cancelled, terminating.")
    elif type in [-1, 10]:
        if type == 10:
            logger.info(f"[EXIT HANDLER] Job timed out, saving checkpoint.")
        else:
            logger.info(f"[EXIT HANDLER] Error during training encountered, saving checkpoint.")
        os.makedirs(checkpoint_path, exist_ok=True)
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'training_step': training_step,
        }, os.path.join(checkpoint_path, f"checkpoint_{JOBID}.ckpt"))
        logger.info(f"[EXIT HANDLER] Checkpoint saved at step {training_step}")

        if type == 10:
            ret = os.system(f"sbatch {WORKDIR}/train.sh {JOBID}")
            if ret != 0:
                logger.info(f"[EXIT HANDLER] Failed to requeue job {JOBID}.")
            else:
                logger.info(f"[EXIT HANDLER] sbatch requeued, new job will load the last checkpoint")
    else:
        logger.info(f"[EXIT HANDLER] Unknown exit signal {type}, terminating.")


def catch_SIG_exception(signum, frame):
    """
    Generic exit handler for both SIGTERM, SIGUSR and exceptions, which just raises and error.
    """
    raise Exception("Exception", signum)


@contextmanager
def set_default_dtype(dtype: torch.dtype):
    """
    Context manager to set torch's default dtype.
    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="/capstor/store/cscs/ethz/large-sc/datasets/train_data.parquet",
        help="Path to a parquet file containing a 'text' column with documents (`str`)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=f"{WORKDIR}/checkpoints",
        help="Path to a checkpoint file to save the model and optimizer state dicts",
    )
    parser.add_argument(
        "--checkpoint-id",
        type=str,
        default="",
        help="Path to a checkpoint file to save the model and optimizer state dicts",
    )
    parser.add_argument(
        "--tokenizer-name-or-path",
        type=str,
        default="unsloth/Mistral-Nemo-Base-2407-bnb-4bit",
        help="A path to a directory containing vocabulary files required by the tokenizer or the model id of a predefined tokenizer hosted inside a model repo on the Hugging Face Hub.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--fused-optimizer",
        action='store_true',
        help="Set to fuse the optimizer for increased performance or not"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--logging-frequency",
        type=int,
        default=5,
        help="Log every `--logging-frequency` steps"
    )

    parser.add_argument(
        "--grad-max-norm",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--model-dtype",
        type=str,
        default="bf16",
        help="Model dtype for parameters, gradients and optimizer states. Default: bf16",
    )
    parser.add_argument(
        "--compile",
        action='store_true',
        help="Set to compile the model with `torch.compile`"
    )
    parser.add_argument(
        "--raise-error",
        action='store_true',
        help="Set to raise an error in the training loop at the error_step parameter",
    )
    parser.add_argument(
        "--error-step",
        type=int,
        default=100,
        help="Step at which to raise an error if --raise-error is set",
    )
    args = parser.parse_args()
    return args