# Large Scale AI Project Report

*Danilo Dordevic, danilo.dordevic@sed.ethz.ch*

*Vukasin Bozic, vukasin.bozic@ethz.ch*

## Problem Statement

Distributed training on large-scale clusters using Slurm as the workload manager is prone to interruptions, many of which arise from unhandled exceptions. A common issue is the **time limit exception**, which occurs when a submitted job exceeds the time requested at submission. For example, if a training run actually takes several days to complete, but the user underestimates this and allocates only a few hours, the job will be terminated abruptly by the scheduler once the time limit is reached. This often results in the **loss of unsaved model weights**, leading to significant computational and energy waste.

Another source of failure is **runtime errors in the Python code**, which can halt training unexpectedly and similarly result in the loss of progress. These errors are particularly problematic when checkpoints are not frequently or properly saved.

## Solution

To mitigate the effects of these interruptions, we implemented a robust checkpointing and error-handling system that addresses three types of interruptions:

- Slurm timeout (time limit exceeded)
- Python runtime exceptions
- Manual termination via scancel

### **2.1 Handling Slurm Timeout Errors**

Slurm allows users to define signal-based job preemption. We configure Slurm to send the USR1 signal **two minutes before** the job’s time limit expires. This lead time, empirically determined based on model size and system I/O speed, is sufficient to trigger a graceful shutdown. For illustration, we set the job time limit to **six minutes** in our experiments, with **two minutes** of lead time.

Within the training script, a signal handler captures the USR1 signal. This is done by first initializing the **signal handlers**: `signal.signal(signal.SIGUSR1, catch_SIG_exception)`, done on line 90 in the [train.py](http://train.py) file. Upon receiving it, the system:

- Saves the current model parameters (checkpoint)
- Saves the states of the optimizer, data loader, and learning rate scheduler
- Submits a new job instance using the same script, continuing from the last checkpoint

This enables **seamless resumption** and continuity of chained training across multiple jobs.

The training loop is encapsulated in a `try` block to capture all exception types discussed earlier. Each exception is associated with a specific `error_type`, represented as an integer, which is passed to the `handle_exit` function (defined on line 65 of the [utils.py](http://utils.py) file). This allows the function to accurately identify the source of the interruption and respond appropriately, for example, by saving checkpoints, resubmitting the job, or just exiting cleanly.

### **2.2 Handling Python Code Errors**

Python exceptions unrelated to timeouts are also captured. In such cases, similar to [**2.1 Handling Slurm Timeout Errors**](https://www.notion.so/2-1-Handling-Slurm-Timeout-Errors-1f5ceedefc8a8032bbb1c7c3760eb0fe?pvs=21), the system:

- Saves the current state of the model, optimizer, data loader, and scheduler

However, **automatic resubmission is not triggered**, since the error likely originates from a bug in the source code and would persist in subsequent runs, making resubmission attempts pointless.

### **2.3 Handling Manual Job Termination (scancel)**

When the job is terminated manually using the `scancel` command, we detect this event, log it for auditing purposes, and **exit immediately** without saving any state. To detect, we need to set up the signal handler (done on line 91 of the [train.py](http://train.py) file): `signal.signal(signal.SIGTERM, catch_SIG_exception)`.

Since this action is initiated by the user, we assume it is intentional and final.

### 2.4 Running the code

To reproduce the results and log files presented in the report, one can simply run the bash script:

`sbatch train.sh`. This command initiates a new training run. The script accepts an optional argument `checkpoint_id`, which specifies the identifier of a previous job whose checkpoint should be used to resume training. For example: `bash [train.sh](http://train.sh/) --checkpoint_id=234`. In every subsequent job, which is automatically scheduled in the timeout handler, this argument will be internally passed, which represents the `JOBID` of a previous job that was timeouted and the name under which the checkpoint of that job was saved. Then, that checkpoint will be loaded, and the training will be continued.

To test the Python exception handling, we set up an easy interface to control that directly from the [train.sh](http://train.sh) script. By modifying the `TRAINING_CMD` variable, one can pass arguments `--raise_error` and `--error_step`, to intentionally raise a dummy error which is going to trigger the error handling exactly at `training_step == error_step`, as observed in  [output_444671.out](https://www.notion.so/Large-Scale-AI-Project-Report-1f5ceedefc8a8068b18ce61efad8c80f?pvs=21). For example, the TRAINING_CMD in train.sh can be modified as follows:

```bash
TRAINING_CMD="--sequence-length 2048 \
                            --batch-size 1 \
                            --learning-rate 5e-5 \
                            --lr-warmup-steps 100 \
                            --training-steps 1400 \
                            --raise-error \
                            --error-step 1200"
```

Finally, to test the manual cancel resolution, you can just run `scancel` command on a running job to have the job cancelled and training stopped, without any checkpointing or handling of the interrupted training.

### **2.5 Results**

The file [output_444691.out](https://www.notion.so/Large-Scale-AI-Project-Report-1f5ceedefc8a8068b18ce61efad8c80f?pvs=21) illustrates the behavior of the system upon manual termination of a job. The final log entry - `[EXIT HANDLER] Job cancelled, terminating.` clearly indicates that the job was explicitly cancelled by the user, and the termination was handled as expected.

In contrast, the files [output_444664.out](https://www.notion.so/Large-Scale-AI-Project-Report-1f5ceedefc8a8068b18ce61efad8c80f?pvs=21)  and [output_444671.out](https://www.notion.so/Large-Scale-AI-Project-Report-1f5ceedefc8a8068b18ce61efad8c80f?pvs=21) illustrate how training is gracefully interrupted due to a timeout and subsequently resumed in a new job submission, without any loss of progress, until we artificially raise a dummy error in Python code, which saves the last submission and gracefully exits.

In [output_444664.out](https://www.notion.so/Large-Scale-AI-Project-Report-1f5ceedefc8a8068b18ce61efad8c80f?pvs=21), the exit handler logs the event with the message: `[EXIT HANDLER] Job timed out, saving checkpoint.` , followed by confirmation that the checkpoint was successfully saved - `[EXIT HANDLER] Checkpoint saved at step 427` , and that the training will resume in a new job submission: `[EXIT HANDLER] sbatch requeued, new job will load the last checkpoint`.

The subsequent file [output_444671.out](https://www.notion.so/Large-Scale-AI-Project-Report-1f5ceedefc8a8068b18ce61efad8c80f?pvs=21), confirms that the training process resumes correctly from the previously saved state, as indicated by the log entry:  `Resuming training from training_step 427` . In the same run, a simulated Python error is introduced to verify the robustness of the handler. The log output confirms appropriate handling of the exception: `[EXIT HANDLER] Error during training encountered, saving checkpoint.` This ensures that even in the presence of unexpected failures, the system reliably preserves training progress.

## Conclusion

Dealing with exceptions gracefully in large-scale model training becomes increasingly important as training duration, model complexity, and computational costs grow, along with the stakes of failure. In this work, we introduced a practical and extensible solution tailored for Slurm-managed clusters, aimed at minimizing disruptions caused by timeouts, runtime errors, and user interruptions. Our method is particularly well-suited for large-scale language model training, where uninterrupted execution is critical, and has demonstrated both effectiveness and reliability in preserving training progress under various failure scenarios.

## Appendix

[output_444664.out](Large%20Scale%20AI%20Project%20Report%201f5ceedefc8a8068b18ce61efad8c80f/output_444664.out)

[output_444671.out](Large%20Scale%20AI%20Project%20Report%201f5ceedefc8a8068b18ce61efad8c80f/output_444671.out)

[output_444691.out](Large%20Scale%20AI%20Project%20Report%201f5ceedefc8a8068b18ce61efad8c80f/output_444691.out)