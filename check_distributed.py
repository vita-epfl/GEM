import torch
import torch.distributed as dist
import os


def check_distributed_training():
    # Check if PyTorch is available
    if not torch.cuda.is_available():
        print(
            "CUDA is not available. Distributed training is typically done with GPUs."
        )
        return False

    # Check if NCCL backend is available (for multi-GPU training)
    if not dist.is_nccl_available():
        print("NCCL backend is not available.")
        return False

    # Check if your system has more than one GPU for multi-GPU training
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(
            f"Only {num_gpus} GPU(s) detected. Distributed training usually requires at least 2 GPUs."
        )
        return False
    else:
        print(f"{num_gpus} GPU(s) detected, suitable for distributed training.")

    # Check if 'torch.distributed' package is properly installed and can be initialized
    try:
        dist.init_process_group(
            backend="nccl", init_method="tcp://127.0.0.1:23456", rank=0, world_size=1
        )
        print("Distributed package initialized successfully.")
    except Exception as e:
        print(f"Error initializing distributed package: {e}")
        return False

    return True


if __name__ == "__main__":
    if check_distributed_training():
        print("Distributed training is possible on this machine.")
    else:
        print("Distributed training is not possible on this machine.")
