# Feature-Level De-raining RT-DETR

## Quick Reference

| Component | Input | Output | Checkpoint |
|-----------|-------|--------|------------|
| RT-DETR | [0,1] 640×640 | Detections | `PekingU/rtdetr_r18vd` |
| Feature Derain | Multi-scale features | Enhanced features | `outputs_feature_derain/feature_derain_best.pt` |
| SPDNet (baseline) | [0,255] | [0,255] | `model_spa.pt` |

## ⚠️ Critical Integration Pattern

```python
# Feature De-rain (main approach - operates on features, not pixels)
from utils.feature_derain import FeatureDerainRTDETR

model = FeatureDerainRTDETR(rtdetr, derain_type="multiscale")
outputs = model(pixel_values=images)  # Handles de-raining internally

# SPDNet + RT-DETR baseline (pixel-level, for comparison only)
spdnet_input = pixel_values * 255.0      # Scale UP for SPDNet
derain_output = spdnet_model(spdnet_input)
clean_images = torch.clamp(derain_output / 255.0, 0, 1)  # Scale DOWN for RT-DETR
```

## Dataset Paths
- **Clean COCO**: `E:\Python\DLCV\dataset\coco`
- **Rainy COCO**: `E:\Python\DLCV\dataset\coco_rain`
- **SPDNet external**: `E:\Python\DLCV\SPDNet`

## Performance Benchmarks

| Model | mAP | AP50 | Latency | Notes |
|-------|-----|------|---------|-------|
| Vanilla RT-DETR | 34.1% | 48.3% | 25ms | Baseline |
| SPDNet + RT-DETR | 33.7% | 48.3% | 205ms | Pixel-level |
| **Feature De-rain** | **34.9%** | **49.1%** | **28ms** | Ours ✅ |

## Terminal Setup
```powershell
& E:\Python\DLCV\.venv\Scripts\Activate.ps1  # ALWAYS activate first
```

## Key Scripts

| Script | Purpose |
|--------|---------|
| `Training_FeatureDerain.py` | Train feature de-rain module |
| `Eval_FeatureDerain.py` | Evaluate on COCO-Rain |
| `Eval_FeatureDerain_MixedRain.py` | Compare all models on MixedRain |

## Common Bugs

### Rain Labeling
```python
# ❌ WRONG: if 'rain' in path  (matches 'train2017')
# ✅ CORRECT: if 'coco_rain' in path
```

### OOM Fix
```python
BATCH_SIZE = 8; GRADIENT_ACCUMULATION_STEPS = 2
```

## Utils Reference

| Module | Key Functions |
|--------|---------------|
| `feature_derain.py` | `FeatureDerainRTDETR`, `create_feature_derain_rtdetr()` |
| `data_utils.py` | `load_datasets()`, `AugmentedDetectionDataset` |
| `eval_utils.py` | `generate_predictions()`, `evaluate_coco()` |
| `spdnet_utils.py` | `load_spdnet_model()` (CUDA required, baseline only) |

## Training Config
```python
PERCENT_DATASET = 100   # Use 100% for full training
COCO_RATIO = 0.3        # 30% clean images
RAIN_RATIO = 0.7        # 70% rainy images
PHASE1_EPOCHS = 5       # De-rain module only (frozen RT-DETR)
PHASE2_EPOCHS = 10      # Joint fine-tuning
BATCH_SIZE = 16
```

## Architecture
Feature De-rain operates at backbone output level:
```
Rainy Image → RT-DETR Backbone → Feature De-rain Module → Encoder → Decoder → Detections
                                 (Rain mask + CBAM attention)
```

**Key insight**: 10x faster than pixel-level de-raining by operating on smaller feature tensors.

## GPU: 8GB min (inference), 16GB+ (training)

## Debugging
- **OOM**: Reduce BATCH_SIZE, increase GRADIENT_ACCUMULATION_STEPS
- **Low mAP**: Ensure proper de-rain module loading from checkpoint
- **Slow eval**: Use PERCENT_DATASET = 5 for quick tests
