import os
import time


CONFIG = {
    "cuda_visible_devices": "0",
    "device": "cuda:0",
    "hold_seconds": 3600.0,
    "matrix_size": 12288,
    "dtype": "float16",
    "reserve_gb": 70.0,
    "sleep_ms": 0.0,
    "report_every_seconds": 5.0,
    "seed": 42,
}


os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(CONFIG["cuda_visible_devices"]))

import torch


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def format_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GiB"


def report_cuda_memory(device):
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    used_bytes = total_bytes - free_bytes
    print(
        "[cuda_mem] "
        f"device={device} "
        f"used={format_gib(used_bytes)} "
        f"free={format_gib(free_bytes)} "
        f"total={format_gib(total_bytes)}"
    )


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in the current environment.")

    device = torch.device(CONFIG["device"])
    dtype = DTYPE_MAP[CONFIG["dtype"]]
    hold_seconds = float(CONFIG["hold_seconds"])
    n = int(CONFIG["matrix_size"])

    torch.manual_seed(int(CONFIG["seed"]))
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True

    print(
        "[run_transformer] "
        f"cuda_visible_devices={CONFIG['cuda_visible_devices']} "
        f"device={device} "
        f"matrix_size={n} "
        f"dtype={CONFIG['dtype']} "
        f"hold_seconds={hold_seconds}"
    )

    a = torch.randn((n, n), device=device, dtype=dtype)
    b = torch.randn((n, n), device=device, dtype=dtype)
    c = torch.empty((n, n), device=device, dtype=dtype)

    reserve_buf = None
    if float(CONFIG["reserve_gb"]) > 0:
        reserve_bytes = int(float(CONFIG["reserve_gb"]) * (1024 ** 3))
        reserve_buf = torch.empty((reserve_bytes,), device=device, dtype=torch.uint8)
        reserve_buf.zero_()
        print(f"[run_transformer] reserved_extra_buffer={format_gib(reserve_bytes)}")

    report_cuda_memory(device)

    deadline = None if hold_seconds <= 0 else time.time() + hold_seconds
    report_every = max(float(CONFIG["report_every_seconds"]), 0.1)
    sleep_s = max(float(CONFIG["sleep_ms"]), 0.0) / 1000.0
    last_report = time.time()
    start_time = last_report
    iterations = 0
    flops_per_iter = 2.0 * (n ** 3)

    print("[run_transformer] running. Press Ctrl+C to stop.")
    try:
        with torch.inference_mode():
            while True:
                if deadline is not None and time.time() >= deadline:
                    break

                torch.mm(a, b, out=c)
                torch.tanh_(c)
                a, b, c = b, c, a
                iterations += 1

                if sleep_s > 0:
                    torch.cuda.synchronize(device)
                    time.sleep(sleep_s)

                now = time.time()
                if now - last_report >= report_every:
                    torch.cuda.synchronize(device)
                    elapsed = now - start_time
                    avg_tflops = (iterations * flops_per_iter) / max(elapsed, 1e-6) / 1e12
                    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                    print(
                        "[run_transformer] "
                        f"iters={iterations} "
                        f"avg_tflops={avg_tflops:.2f} "
                        f"free={format_gib(free_bytes)} "
                        f"used={format_gib(total_bytes - free_bytes)}"
                    )
                    last_report = now
    except KeyboardInterrupt:
        print("[run_transformer] interrupted by user")

    torch.cuda.synchronize(device)
    total_elapsed = time.time() - start_time
    avg_tflops = (iterations * flops_per_iter) / max(total_elapsed, 1e-6) / 1e12
    print(
        "[run_transformer] "
        f"finished iters={iterations} "
        f"elapsed_s={total_elapsed:.2f} "
        f"avg_tflops={avg_tflops:.2f}"
    )
    del reserve_buf


if __name__ == "__main__":
    main()
