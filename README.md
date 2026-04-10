# Matrix Cost Visualizer

Lightweight FastAPI app that serves a single-page frontend for exploring compute, communication, and memory behavior in parallel linear algebra workloads.

## Run locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the app:
   ```bash
   uvicorn main:app --reload
   ```
3. Open:
   - http://127.0.0.1:8000

## Notes

- All cost calculations are done in the browser (`static/app.js`).
- FastAPI is only used to serve static files.
- Includes operation presets for `GEMM`, `GEMV`, and `Gram (A^T A)`.
- Users can set matrix shapes for `A` and `B`, data share of `A/B` per processor, and processor count `p`.
- App auto-derives output matrix `C` and updates FLOPs, communication cost, and memory cost in real time.
- Visualization uses a single roofline-derived time plot vs arithmetic intensity.
- Built-in profiling shows frontend calculation and chart update timings per interaction.
- Graphs use arithmetic intensity (`FLOPs / bytes moved`) as the common x-axis.
- Memory cost is modeled as time: `bytes moved / bandwidth + latency terms`.
- Throughput metrics are shown in `GFLOP/s`.
- Includes implementation profiles for `NumPy`, `CUDA Kernel`, and `cuBLAS`.
- Includes GPU presets (`NVIDIA Hopper H100`, `NVIDIA Blackwell B200`, `A100`, `RTX 4090`) plus `Custom`.
- Roofline time model curves:
  - `T_compute = FLOPs / P_peak` (horizontal)
  - `T_memory = FLOPs / (AI * BW)` (decreasing)
  - `T(AI) = max(T_compute, T_memory)` with knee at `AI = P_peak / BW`
- Baseline GEMM formulas used in notes:
  - Compute cost: `2lmn` FLOPs
  - Memory movement: `lm + mn + ln` (elements, then scaled by bytes/element)
  - Memory cost (time): `memory movement / bandwidth + latency`
  - Arithmetic intensity: `AI = compute / memory movement`
  - Square GEMM scaling: for `l = m = n`, `AI = (2n^3)/(3n^2) = 2n/3` (grows linearly with `n`)
- Computation pipeline in code:
  - Derive baseline `l, m, n` from selected operation (`GEMM`, `GEMV`, `Gram`) and compute baseline terms first.
  - Expand baseline with matrix-share and algorithm effects:
    - Share-adjusted movement: `(1-shareA)lm + (1-shareB)mn + ln`
    - Naive/tiled overhead terms layered on top of baseline movement
  - Apply hardware/program limits to map FLOPs and movement into execution time.
