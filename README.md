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
- Visualization is split across separate canvases: compute, communication, memory, and roofline.
- Built-in profiling shows frontend calculation and chart update timings per interaction.
