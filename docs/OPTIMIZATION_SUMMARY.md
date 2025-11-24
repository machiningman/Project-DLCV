# Optimization Summary & Changelog

## 🚀 Performance Optimizations (November 24, 2025)

### 1. Training Speedup (~3-4x faster)
- **Frozen SPDNet**: The de-raining module (SPDNet) is now kept **frozen** throughout the entire training process. It acts as a fixed preprocessing filter.
  - *Benefit*: Saves ~30-40% VRAM and computation.
  - *Benefit*: Enables safe usage of FP16 (Mixed Precision).
- **FP16 Enabled**: Switched `FP16 = True`.
  - *Benefit*: 2-3x speedup on Tensor Cores.
- **Batch Size Increased**: Increased from `4` to `8`.
  - *Benefit*: Better gradient estimates and hardware utilization.
- **Gradient Accumulation Reduced**: Reduced from `4` to `2`.
  - *Benefit*: Maintains effective batch size of 16 (8x2) while reducing overhead.
- **Removed Blocking Launch**: Removed `CUDA_LAUNCH_BLOCKING=1`.
  - *Benefit*: Restores asynchronous GPU execution (fixed the 67s/it slowdown).

### 2. Training Strategy Refinement
- **Simplified to 2 Phases**:
  - **Phase 1 (Epochs 1-3)**: Train Detection Head only (Backbone & SPDNet frozen).
  - **Phase 2 (Epochs 4-12)**: Fine-tune RT-DETR Backbone + Head (SPDNet frozen).
- **Fixed Data Sampling**:
  - Removed hardcoded 20% rain sampler in `utils/training_utils.py`.
  - Now respects the `RAIN_RATIO = 0.5` setting in the main script (50/50 split).

### 3. Stability & Bug Fixes
- **Fixed Matplotlib Worker Warnings**:
  - Removed `matplotlib` imports from `Training_conditional.py` (replaced plots with CSV/JSON logging).
  - Prevented `eval_utils` from being imported in `utils/__init__.py`.
  - Added `warnings.filterwarnings` to suppress internal warnings from the `supervision` library.
- **Fixed OpenCV Threading**:
  - Added `cv2.setNumThreads(0)` to prevent worker deadlocks.

## 📂 Project Restructuring

### Main Directory (Active Files)
- `Training_conditional.py`: Main training script (Optimized).
- `Eval_conditional.py`: Evaluation script.
- `Pretrain_rain_detector.py`: Rain detector pre-training script.

### Archive (`/archive`)
- Legacy training scripts (`Training.py`, `Training_integrated.py`).
- Old notebooks and experiments.
- Unused utility scripts.

### Documentation (`/docs`)
- Project goals, guides, and reference markdown files.
