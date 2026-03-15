# Running autoresearch-at-home on Apple Silicon (MPS)

This branch adds support for running autoresearch training on Apple Silicon Macs using PyTorch's MPS (Metal Performance Shaders) backend.

## What Changed

### train.py
- **Device auto-detection**: Checks for CUDA first, falls back to MPS, then CPU
- **Flash Attention 3 → SDPA fallback**: FA3 is CUDA-only; uses `F.scaled_dot_product_attention` on MPS
- **bfloat16 → float16**: MPS doesn't support bfloat16 autocast; auto-selects float16 on MPS
- **Disabled `torch.compile` on MPS**: Causes OOM with large intermediate tensors on unified memory
- **Reduced `DEVICE_BATCH_SIZE`**: 128 (CUDA default) OOMs on 16GB; uses 16 on MPS
- **`torch.cuda.synchronize()` → `torch.mps.synchronize()`**
- **`torch.cuda.max_memory_allocated()` → graceful skip on MPS** (not supported)
- **MFU calculation**: Uses estimated M4 FP16 peak FLOPS (4.0 TFLOPS) for reporting

### prepare.py
- **Device detection**: `gpu_buffer` uses MPS device when available
- **Disabled `pin_memory`**: MPS doesn't support pinned memory
- **Disabled `non_blocking` transfers**: Not supported on MPS

## Benchmark: Mac mini M4 (10-core GPU, 16GB unified memory)

| Metric | Value |
|--------|-------|
| Steps completed | 12 |
| Loss | 9.01 → 7.84 |
| Throughput | ~250 tok/sec |
| MFU | ~1.5–2.5% |
| Wall time per step | ~30 min |
| Batch size | 16 |
| Autocast dtype | float16 |
| torch.compile | disabled |

### Comparison with CUDA

| | Mac mini M4 | H100 (typical) |
|---|---|---|
| tok/sec | ~250 | ~500,000+ |
| Time per step | ~30 min | ~1–2 sec |
| MFU | ~2% | ~30–50% |

The M4 is roughly **1000x slower** than an H100 for this workload. The bottlenecks are:
1. No Flash Attention 3 (using PyTorch SDPA fallback — functional but slower)
2. No `torch.compile` (MPS backend doesn't support it well yet)
3. Much less raw compute (10 GPU cores vs 80 SMs)

## Is it useful?

As a swarm contributor? Not yet — an M4 can't keep up with CUDA nodes on 5-minute experiment cycles.

But this patch has value for:
- **Local development & testing** — iterate on model code without needing a GPU server
- **M4 Max / M4 Ultra owners** — significantly more GPU cores + memory; could be viable
- **Future MPS improvements** — as PyTorch's MPS backend matures (torch.compile support, better kernels), the gap will shrink
- **Reference implementation** — shows the minimal changes needed for Apple Silicon support

## Setup

```bash
# Clone and install
git clone https://github.com/bdecrem/autoresearch-at-home.git
cd autoresearch-at-home
git checkout apple-silicon-mps
uv sync

# Prepare data
uv run python prepare.py

# Run training
uv run python train.py
```

Device detection is automatic — no flags needed. If MPS is available, it uses MPS. If CUDA is available, it uses CUDA.

## macOS Version

Tested on macOS 26.1 (Tahoe), Mac mini M4 (10-core GPU, 16GB RAM), Python 3.12, PyTorch 2.6+.
