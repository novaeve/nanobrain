# Environment

- Python: tested with 3.10.20
- PyTorch: tested with 2.1.2+cu121
- CUDA runtime used by PyTorch: 12.1
- NVIDIA driver observed during cleanup: 570.124.06
- System CUDA reported by `nvidia-smi`: 12.8

Install with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

GPU training was developed on CUDA-enabled PyTorch. CPU-only installs can import the package and run lightweight checks, but full training is expected to be slow.
