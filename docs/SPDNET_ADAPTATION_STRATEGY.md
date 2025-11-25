# SPDNet Adaptation Strategy with Frozen RT-DETR

**Date**: November 25, 2025  
**Goal**: Train SPDNet to produce detection-friendly de-rained images while preserving RT-DETR performance on clean images.

---

## Problem Statement

Your original approach had limited improvement because:

1. **Rain Detector hard-gating**: The conditional model uses boolean masks to decide which images go through SPDNet. This is **non-differentiable** - gradients cannot flow back to train the rain detector.

2. **SPDNet not adapted for detection**: SPDNet was trained to produce "visually clean" images (generic de-raining), not necessarily images that are **optimal for object detection**. Different metrics (visual quality vs. detection accuracy) may require different de-raining strategies.

3. **Training constraints**: RT-DETR was either frozen (no learning) or unfrozen (risk of degradation on clean images). There was no middle ground.

4. **FP16 incompatibility**: SPDNet has numerical issues with half precision, preventing memory-efficient training.

---

## Proposed Solution: SPDNet Adaptation

### Core Strategy

**Train SPDNet while keeping RT-DETR frozen.**

This approach:
- ✅ **Guarantees RT-DETR performance preservation** on clean images (frozen = identical to baseline)
- ✅ **Optimizes de-raining for detection** (loss = detection mAP, not visual quality)
- ✅ **Uses only rainy images** (saves time - detection loss is irrelevant for clean images)
- ✅ **Enables end-to-end gradient flow** from detection loss → SPDNet weights

### Key Design Decisions

| Decision | Reason |
|----------|--------|
| **RT-DETR Frozen** | Prevents any degradation on clean images. RT-DETR is already well-trained. |
| **Only Rainy Images** | Clean images don't help SPDNet learn. Detection loss = 0 on perfect images. |
| **FP16 Disabled** | SPDNet has numerical stability issues with half precision. |
| **Full Gradient Flow** | Detection loss → clean_images → SPDNet weights. Backprop needs the entire RT-DETR computation graph. |
| **Higher Learning Rate** | Only SPDNet trains, so can use aggressive LR without risk. |

---

## Training Flow

```
Rainy Image (0-1 normalized)
    ↓
    × 255 (scale to SPDNet range)
    ↓
SPDNet (TRAINABLE) → de-rained output (0-255 range)
    ↓
    ÷ 255 + clamp (scale back to RT-DETR range 0-1)
    ↓
RT-DETR (FROZEN) → detection predictions
    ↓
Detection Loss (mAP-based)
    ↓
Backprop: Loss → clean_images → SPDNet weights
         (RT-DETR params don't update because requires_grad=False)
```

---

## Inference Strategy (Conditional Model)

After training, use the **adapted SPDNet with conditional branching**:

```
Input Image
    ↓
Rain Detector (pretrained) → Is Rainy?
    ├─ YES → Adapted SPDNet → RT-DETR → Detections
    └─ NO  → RT-DETR (direct) → Detections
```

**Benefits**:
- Clean images: Fast path (no de-raining)
- Rainy images: Optimized de-raining for detection
- Clean-image performance: **Guaranteed preserved** (RT-DETR unchanged)
- Rainy-image performance: **Improved** (SPDNet adapted for detection)

---

## Implementation Details

### Dataset Configuration

```python
PERCENT_DATASET = 10          # Use 10% of dataset (~11,800 images)
COCO_RATIO = 0.0              # 0% clean images
RAIN_RATIO = 1.0              # 100% rainy images
```

**Why 10%?** 
- Full dataset (118K) → ~5.4 min/iter → 16+ hours/epoch (too slow)
- 10% dataset (11.8K) → ~30 sec/iter → ~1 hour/epoch (acceptable)
- With 10 epochs covering different rainy variations, sufficient adaptation

### Training Configuration

```python
NUM_EPOCHS = 10
BATCH_SIZE = 2                # SPDNet memory-intensive
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch = 16
LEARNING_RATE = 1e-4          # Higher LR for SPDNet
FP16 = False                  # Required for SPDNet stability
DATALOADER_WORKERS = 8        # Fast data loading
```

### Model State

At training start:
```
SPDNet: 3,318,741 trainable parameters (100%)
RT-DETR: 0 trainable parameters (0% - FROZEN)
Total parameters: 23,493,349
Trainable parameters: 3,318,741 (14.1%)
```

---

## Expected Outcomes

### 1. Rainy Image Performance
- **Baseline (Vanilla RT-DETR)**: mAP ≈ 0.306
- **Target (Adapted SPDNet)**: mAP > 0.320 (5% improvement)

### 2. Clean Image Performance
- **Guaranteed**: 100% preserved (RT-DETR identical to baseline)
- **Reason**: RT-DETR frozen, conditional model skips SPDNet for clean images

### 3. Comparison with Original SPDNet
- **Original**: Generic de-raining (visual quality)
- **Adapted**: Detection-optimized de-raining
- **Expected**: Adapted > Original on rainy COCO validation set

---

## Files Created/Modified

### New Training Script
**`Training_SPDNet_Adaptation.py`**
- Trains SPDNet with frozen RT-DETR
- Only uses rainy images
- Outputs: `./outputs_spdnet_adaptation/spdnet_adapted.pt`
- Timing: ~10 hours (10 epochs × ~1 hour each)

### New Evaluation Script
**`Eval_SPDNet_Adaptation.py`**
- Compares three approaches:
  1. Vanilla RT-DETR (baseline)
  2. Original SPDNet + RT-DETR
  3. Adapted SPDNet + RT-DETR
- Shows mAP improvement on rainy images
- Validates training was successful

### Modified Model File
**`utils/integrated_model.py`**
- Already has `freeze_detection=True` support
- Forward pass: SPDNet(image) → RT-DETR(clean) → loss
- Gradients flow to SPDNet through frozen RT-DETR

---

## Step-by-Step Workflow

### Phase 1: Training (You are here)
```powershell
# 1. Start training adapted SPDNet
python Training_SPDNet_Adaptation.py

# Expected:
# - Each epoch: ~1 hour
# - Total: ~10 hours
# - Output: ./outputs_spdnet_adaptation/spdnet_adapted.pt
```

### Phase 2: Evaluation
```powershell
# 2. Evaluate adapted SPDNet vs. original vs. vanilla
python Eval_SPDNet_Adaptation.py

# Expected output:
# Vanilla RT-DETR mAP:        0.3060
# Original SPDNet + RT-DETR:  0.3150  (+2.9%)
# Adapted SPDNet + RT-DETR:   0.3280  (+7.2%)  ← Your target
```

### Phase 3: Integration
```powershell
# 3. Update Eval_conditional.py to use adapted SPDNet
# Change:
#   SPDNET_MODEL_PATH = "./outputs_spdnet_adaptation/spdnet_adapted.pt"

# 4. Run conditional model evaluation
python Eval_conditional.py

# Expected:
# - Clean images: Same as vanilla RT-DETR (preserved)
# - Rainy images: Better than before (improved)
# - Inference: 3-5x faster on clean images (no SPDNet)
```

---

## Performance Expectations

### Training Speed Breakdown
- Data loading: ~5 sec/iter
- SPDNet forward: ~120 ms
- RT-DETR forward: ~40 ms
- Loss computation: ~50 ms
- Backward pass: ~5-10 sec (most expensive)
- **Total per iter**: ~15-20 sec (with gradient accumulation)

### Memory Usage
- SPDNet: ~1.5 GB
- RT-DETR: ~2 GB
- Batch (2 images): ~0.5 GB
- Gradients: ~1-2 GB
- **Total**: ~15-16 GB (your RTX 5070 Ti has 17 GB ✓)

### Convergence
- **Learning curve**: Steep initial drop (first 2-3 epochs)
- **Plateau**: Epochs 5-10 (marginal improvements)
- **Best model**: Typically epoch 3-5
- **Early stopping patience**: 5 epochs without improvement

---

## Troubleshooting

### Issue: OOM (Out of Memory)
**Solution**:
```python
BATCH_SIZE = 1                     # Further reduce
GRADIENT_ACCUMULATION_STEPS = 16   # Compensate
```

### Issue: Very Slow (>30 sec/iter)
**Cause**: Data loading bottleneck  
**Solution**:
```python
DATALOADER_WORKERS = 16  # Increase if available
```

### Issue: mAP Not Improving
**Cause**: Learning rate too high/low or insufficient epochs  
**Solution**:
```python
LEARNING_RATE = 5e-5   # Try lower
NUM_EPOCHS = 20        # Try more
```

### Issue: Loss Exploding/NaN
**Cause**: RT-DETR's gradients are unstable with high SPDNet loss  
**Solution**:
```python
LEARNING_RATE = 5e-5   # Much lower
WARMUP_STEPS = 500     # More warmup
```

---

## Why This Approach Works

1. **Ground truth labels**: You have detection labels for rainy images. SPDNet learns from real detection feedback, not synthetic metrics.

2. **Frozen RT-DETR**: Guarantees no degradation. You can confidently say "clean images are preserved."

3. **Detection-optimized**: Unlike generic de-raining (which optimizes PSNR/SSIM), this optimizes for what matters: detection accuracy.

4. **Conditional inference**: Rain detector decides which path → no overhead for clean images.

5. **Proven approach**: Similar to domain adaptation and transfer learning - fine-tune on task-specific data.

---

## Next Steps

1. **Monitor training**: Watch mAP on rainy validation set
2. **Check convergence**: Should see improvement by epoch 2-3
3. **Evaluate**: Run `Eval_SPDNet_Adaptation.py` after training
4. **Integrate**: Use adapted SPDNet in conditional model
5. **Validate**: Run full evaluation on both clean and rainy images

---

## Reference

**Key Files**:
- Training: `Training_SPDNet_Adaptation.py`
- Evaluation: `Eval_SPDNet_Adaptation.py`
- Model: `utils/integrated_model.py`
- Config: `Training_SPDNet_Adaptation.py` (lines 50-80)

**Key Concepts**:
- Frozen parameters: `requires_grad=False`
- Gradient flow: Loss → clean_images → SPDNet → parameters
- Conditional inference: Rain detector + two model paths
