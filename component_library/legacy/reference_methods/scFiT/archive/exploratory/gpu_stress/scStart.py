#!/usr/bin/env python3
"""Reserve GPU memory and keep the GPU busy with matrix multiplications."""

import argparse
import signal
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reserve CUDA memory and run matrix multiplication to increase GPU utilization."
    )
    parser.add_argument(
        "-m",
        "--memory-gb",
        type=float,
        required=True,
        help="GPU memory to reserve in GiB, e.g. -m 40 reserves about 40 GiB.",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=int,
        default=0,
        help="GPU index, default: 0.",
    )
    parser.add_argument(
        "-s",
        "--matrix-size",
        type=int,
        default=8192,
        help="Square matrix size used for GEMM, default: 8192.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help="Tensor dtype for matrix multiplication, default: float16.",
    )
    parser.add_argument(
        "--chunk-gb",
        type=float,
        default=1.0,
        help="Allocation chunk size in GiB, default: 1.0.",
    )
    return parser.parse_args()



def reserve_memory(torch, device, memory_gb, chunk_gb):
    target_bytes = int(memory_gb * 1024**3)
    chunk_bytes = int(chunk_gb * 1024**3)
    if target_bytes <= 0:
        raise SystemExit("--memory-gb must be greater than 0")
    if chunk_bytes <= 0:
        raise SystemExit("--chunk-gb must be greater than 0")

    chunks = []
    allocated = 0
    while allocated < target_bytes:
        current_bytes = min(chunk_bytes, target_bytes - allocated)
        elements = current_bytes // 2
        tensor = torch.empty(elements, dtype=torch.float16, device=device)
        tensor.zero_()
        chunks.append(tensor)
        allocated += tensor.numel() * tensor.element_size()
        print(f"reserved {allocated / 1024**3:.2f} GiB / {memory_gb:.2f} GiB", flush=True)
    return chunks


def main():
    args = parse_args()
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    device = torch.device(f"cuda:{args.device}")
    dtype = getattr(torch, args.dtype)
    stop = False

    def handle_signal(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    reserved = reserve_memory(torch, device, args.memory_gb, args.chunk_gb)

    a = torch.randn((args.matrix_size, args.matrix_size), dtype=dtype, device=device)
    b = torch.randn((args.matrix_size, args.matrix_size), dtype=dtype, device=device)
    c = None
    iterations = 0

    print("starting matrix multiplication loop; press Ctrl+C to stop", flush=True)
    while not stop:
        c = a @ b
        iterations += 1
        if iterations % 20 == 0:
            torch.cuda.synchronize(device)
            used_gb = torch.cuda.memory_allocated(device) / 1024**3
            print(f"iterations={iterations}, torch_allocated={used_gb:.2f} GiB", flush=True)

    del c, a, b, reserved
    torch.cuda.empty_cache()
    print("stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
