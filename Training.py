#!/usr/bin/env python3
"""
Training script for RT-DETR object detection model.

This script loads datasets, configures the model, and trains it on COCO dataset.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

# Import utility modules (from utils package)
from utils.data_utils import load_datasets, create_detection_datasets, collate_fn, split_by_domain, check_split_leakage
from utils.model_utils import load_model_and_processor, print_model_info
from utils.training_utils import (
    ObjectDetectionTrainer, 
    compute_metrics, 
    get_training_arguments,
    get_freeze_backbone_callback,
    get_early_stopping_callback
)

# Configuration
COCO_DIR = "E:/Python/DLCV/Project/dataset/coco"
COCO_RAIN_DIR = "E:/Python/DLCV/Project/dataset/coco_rain"
OUTPUT_DIR = "./outputs"
MODEL_NAME = "PekingU/rtdetr_r18vd"
NUM_LABELS = 80  # COCO classes

# Training configuration
PERCENT_DATASET = 100  # Use x% of dataset
COCO_RATIO = 0.9
RAIN_RATIO = 0.1
NUM_EPOCHS = 12
BATCH_SIZE = 16  # Keep batch size at 16 to avoid memory issues
EVAL_BATCH_SIZE = 16  # Keep eval batch size at 16
GRADIENT_ACCUMULATION_STEPS = 1  # Accumulate gradients for effective batch size of 32
LEARNING_RATE = 1e-5
SEED = 42
FP16 = True  # Enable mixed precision for 2x speedup with Tensor Cores
DATALOADER_WORKERS = 16  # Increased from 4 to utilize all CPU cores (20 available)

# Callback configuration
FREEZE_BACKBONE_EPOCHS = 2  # Freeze backbone for first N epochs
EARLY_STOPPING_PATIENCE = 3  # Stop if no improvement for N epochs

# Set environment variable for CUDA
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def main():
    """Main training function"""
    print("=" * 80)
    print("RT-DETR Training Script")
    print("=" * 80)
    # Performance knobs
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
    
    # Load datasets
    print("\nStep 1: Loading datasets...")
    ds_train, ds_valid = load_datasets(
        coco_dir=COCO_DIR,
        coco_rain_dir=COCO_RAIN_DIR,
        coco_ratio=COCO_RATIO,
        rain_ratio=RAIN_RATIO,
        seed=SEED
    )
    # Quick leakage check
    check_split_leakage(ds_train, ds_valid)
    
    # Load model and processor
    print("\nStep 2: Loading model and processor...")
    model, processor = load_model_and_processor(
        model_name=MODEL_NAME,
        num_labels=NUM_LABELS
    )
    print_model_info(model)
    
    # Create augmented datasets
    print("\nStep 3: Creating augmented datasets...")
    train_dataset, valid_dataset = create_detection_datasets(
        ds_train=ds_train,
        ds_valid=ds_valid,
        processor=processor,
        percent_dataset=PERCENT_DATASET
    )
    
    # Get training arguments
    print("\nStep 4: Configuring training arguments...")
    args = get_training_arguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        seed=SEED,
        fp16=FP16,
        dataloader_num_workers=DATALOADER_WORKERS,
        lr_scheduler_type="cosine",
        logging_steps=100
    )
    
    # Create trainer
    print("\nStep 5: Creating trainer...")
    
    # Setup callbacks
    callbacks = [
        get_freeze_backbone_callback(unfreeze_epoch=FREEZE_BACKBONE_EPOCHS),
        get_early_stopping_callback(patience=EARLY_STOPPING_PATIENCE)
    ]
    
    trainer = ObjectDetectionTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=processor,
        data_collator=collate_fn,
        compute_metrics=partial(compute_metrics, processor=processor, threshold=0.05),
        callbacks=callbacks,
    )
    
    # Train model
    print("\nStep 6: Starting training...")
    print("=" * 80)
    trainer.train()
    
    # Save best model
    print("\nStep 7: Saving best model...")
    best = trainer.state.best_model_checkpoint
    if best:
        trainer.save_model(f"{OUTPUT_DIR}/best_from_training")
        print(f"Best model saved to {OUTPUT_DIR}/best_from_training")
    
    # Plot training curves
    print("\nStep 8: Plotting training curves...")
    plot_training_curves(trainer, args)

    # Extra: evaluate on rainy-only validation split
    print("\nStep 8b: Evaluating on rainy-only validation split...")
    valid_clean, valid_rain = split_by_domain(ds_valid)
    _, valid_rain_dataset = create_detection_datasets(valid_clean, valid_rain, processor, percent_dataset=100)
    rain_metrics = trainer.evaluate(eval_dataset=valid_rain_dataset)
    print(f"Rainy-only evaluation: {rain_metrics}")
    
    # Clean up
    print("\nStep 9: Cleaning up...")
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)


def plot_training_curves(trainer, args):
    """Plot training and validation curves"""
    # Initialize arrays
    num_epochs = int(getattr(args, 'num_train_epochs', 0))
    training_loss = np.full(num_epochs, np.nan)
    val_loss = np.full(num_epochs, np.nan)
    val_map = np.full(num_epochs, np.nan)
    
    log_history = trainer.state.log_history
    
    # Extract epoch-level metrics
    for i, entry in enumerate(log_history):
        if 'epoch' not in entry:
            continue
        
        try:
            epoch_val = float(entry['epoch'])
        except Exception:
            continue
        
        if not epoch_val.is_integer():
            continue
        
        epoch_idx = int(epoch_val) - 1
        if epoch_idx < 0 or epoch_idx >= num_epochs:
            continue
        
        if 'eval_map' in entry:
            val_map[epoch_idx] = float(entry.get('eval_map', np.nan))
            val_loss[epoch_idx] = float(entry.get('eval_loss', np.nan))
            
            # Find most recent training loss
            for j in range(i - 1, -1, -1):
                prev = log_history[j]
                if 'loss' in prev:
                    try:
                        training_loss[epoch_idx] = float(prev['loss'])
                        break
                    except Exception:
                        continue
    
    # Trim trailing NaNs
    valid_epochs = ~np.isnan(val_map) | ~np.isnan(training_loss) | ~np.isnan(val_loss)
    if valid_epochs.sum() > 0:
        training_loss = training_loss[valid_epochs]
        val_loss = val_loss[valid_epochs]
        val_map = val_map[valid_epochs]
    
    print(f'Epochs found: {len(val_map)}')
    print(f'training_loss: {training_loss}')
    print(f'val_loss: {val_loss}')
    print(f'val_map: {val_map}')
    
    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss subplot
    if len(training_loss) > 0:
        axs[0].plot(training_loss, label='Training Loss')
    if len(val_loss) > 0:
        axs[0].plot(val_loss, label='Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].set_title('Training and Validation Loss over Epochs')
    axs[0].grid(True)
    
    # mAP subplot
    if len(val_map) > 0:
        axs[1].plot(val_map, label='Validation mAP', color='purple')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('mAP')
    axs[1].legend()
    axs[1].set_title('Validation mAP over Epochs')
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training curves saved to {OUTPUT_DIR}/training_curves.png")


if __name__ == "__main__":
    main()
