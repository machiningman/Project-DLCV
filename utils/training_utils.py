"""Training utilities including custom trainer and metrics computation"""

import torch
import numpy as np
from dataclasses import dataclass
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import Trainer, TrainerCallback, EarlyStoppingCallback
from transformers.image_transforms import center_to_corners_format
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction


@dataclass
class ModelOutput:
    """Model output wrapper for predictions"""
    logits: torch.Tensor
    pred_boxes: torch.Tensor


class ObjectDetectionTrainer(Trainer):
    """Custom Trainer for object detection that properly handles predictions and sampling"""

    def get_train_dataloader(self):
        """Use a domain-balanced sampler when possible (target ~20% rainy per batch)."""
        if self.train_dataset is None:
            return None

        try:
            from torch.utils.data import DataLoader, WeightedRandomSampler

            dataset = self.train_dataset
            # Try to get underlying image paths to detect rainy samples
            paths = None
            if hasattr(dataset, "get_path"):
                paths = [dataset.get_path(i) for i in range(len(dataset))]
            elif hasattr(dataset, "dataset") and hasattr(dataset.dataset, "image_paths"):
                paths = list(getattr(dataset.dataset, "image_paths"))

            if paths:
                weights = []
                for p in paths:
                    p_str = str(p).lower()
                    is_rain = ("coco_rain" in p_str) or ("rain" in p_str)
                    # Target around 20% rainy samples
                    weights.append(0.2 if is_rain else 0.8)

                sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
                return DataLoader(
                    dataset,
                    batch_size=self.args.train_batch_size,
                    sampler=sampler,
                    collate_fn=self.data_collator,
                    num_workers=self.args.dataloader_num_workers,
                    pin_memory=self.args.dataloader_pin_memory,
                    drop_last=True,
                )
        except Exception:
            pass

        # Fallback to default behavior
        return super().get_train_dataloader()

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        model = self.model
        model.eval()
        
        all_logits = []
        all_pred_boxes = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        for step, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            if outputs.loss is not None:
                total_loss += outputs.loss.item()
                num_batches += 1
            
            all_logits.append(outputs.logits.detach().cpu())
            all_pred_boxes.append(outputs.pred_boxes.detach().cpu())
            
            labels = inputs.get("labels")
            if labels is not None:
                for label in labels:
                    if hasattr(label, 'data'):
                        all_labels.append(dict(label.data))
                    elif isinstance(label, dict):
                        all_labels.append(label)
                    else:
                        all_labels.append(dict(label))
        
        logits = torch.cat(all_logits, dim=0)
        pred_boxes = torch.cat(all_pred_boxes, dim=0)
        
        metrics = {}
        if self.compute_metrics is not None:
            eval_pred = EvalPrediction(predictions=(logits, pred_boxes), label_ids=all_labels)
            metrics = self.compute_metrics(eval_pred)
        
        if num_batches > 0:
            metrics["loss"] = total_loss / num_batches
        
        metrics = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}
        
        return EvalLoopOutput(
            predictions=(logits.numpy(), pred_boxes.numpy()),
            label_ids=all_labels,
            metrics=metrics,
            num_samples=len(all_labels)
        )


@torch.no_grad()
def compute_metrics(eval_pred, processor, threshold=0.5):
    """
    Compute mAP metrics for object detection.
    
    Args:
        eval_pred: EvalPrediction object with predictions and labels
        processor: RTDetrImageProcessor for post-processing
        threshold: Confidence threshold for predictions
        
    Returns:
        Dictionary with mAP metric
    """
    predictions, targets = eval_pred.predictions, eval_pred.label_ids
    
    if predictions is None or targets is None:
        return {"map": 0.0}
    
    try:
        if isinstance(predictions, (tuple, list)) and len(predictions) == 2:
            logits, pred_boxes = predictions
            
            if isinstance(logits, np.ndarray):
                logits = torch.from_numpy(logits)
            if isinstance(pred_boxes, np.ndarray):
                pred_boxes = torch.from_numpy(pred_boxes)
        else:
            return {"map": 0.0}
        
        if logits.shape[0] != len(targets):
            return {"map": 0.0}
        
        # Process targets
        image_sizes = []
        post_processed_targets = []
        
        for target in targets:
            if hasattr(target, 'data'):
                target = dict(target.data)
            elif not isinstance(target, dict):
                target = dict(target)
            
            if "orig_size" not in target:
                continue
            
            orig_size = target["orig_size"]
            if isinstance(orig_size, np.ndarray):
                orig_size = torch.from_numpy(orig_size)
            elif not isinstance(orig_size, torch.Tensor):
                orig_size = torch.tensor(orig_size)
            orig_size = orig_size.cpu()
            image_sizes.append(orig_size)
            
            if "boxes" in target and "class_labels" in target and len(target["boxes"]) > 0:
                gt_boxes = target["boxes"]
                gt_labels = target["class_labels"]
                
                if isinstance(gt_boxes, np.ndarray):
                    gt_boxes = torch.from_numpy(gt_boxes)
                elif not isinstance(gt_boxes, torch.Tensor):
                    gt_boxes = torch.tensor(gt_boxes)
                    
                if isinstance(gt_labels, np.ndarray):
                    gt_labels = torch.from_numpy(gt_labels)
                elif not isinstance(gt_labels, torch.Tensor):
                    gt_labels = torch.tensor(gt_labels)
                
                gt_boxes = gt_boxes.cpu()
                gt_labels = gt_labels.cpu()
                
                # Convert from normalized cxcywh to absolute xyxy
                gt_boxes_xyxy = center_to_corners_format(gt_boxes.clone())
                gt_boxes_xyxy[:, 0::2] *= orig_size[1]
                gt_boxes_xyxy[:, 1::2] *= orig_size[0]
                
                # Validate boxes
                valid_mask = (gt_boxes_xyxy[:, 2] > gt_boxes_xyxy[:, 0]) & \
                            (gt_boxes_xyxy[:, 3] > gt_boxes_xyxy[:, 1])
                
                if valid_mask.sum() > 0:
                    post_processed_targets.append({
                        "boxes": gt_boxes_xyxy[valid_mask].cpu(),
                        "labels": gt_labels[valid_mask].cpu()
                    })
                else:
                    post_processed_targets.append({
                        "boxes": torch.empty((0, 4)),
                        "labels": torch.empty((0,), dtype=torch.long)
                    })
            else:
                post_processed_targets.append({
                    "boxes": torch.empty((0, 4)),
                    "labels": torch.empty((0,), dtype=torch.long)
                })
        
        if len(image_sizes) == 0:
            return {"map": 0.0}
        
        image_sizes = torch.stack(image_sizes)
        
        # Post-process predictions
        output = ModelOutput(logits=logits, pred_boxes=pred_boxes)
        post_processed_predictions = processor.post_process_object_detection(
            output, threshold=threshold, target_sizes=image_sizes
        )
        
        # Move predictions to CPU
        for pred in post_processed_predictions:
            pred['boxes'] = pred['boxes'].cpu()
            pred['scores'] = pred['scores'].cpu()
            pred['labels'] = pred['labels'].cpu()
        
        total_gt = sum(len(t['boxes']) for t in post_processed_targets)
        
        if total_gt == 0:
            return {"map": 0.0}
        
        # Compute mAP
        metric = MeanAveragePrecision(box_format="xyxy", class_metrics=False)
        metric.update(post_processed_predictions, post_processed_targets)
        metrics = metric.compute()
        
        map_value = metrics["map"].item() if "map" in metrics else 0.0
        
        return {"map": map_value}
        
    except Exception as e:
        return {"map": 0.0}


def get_training_arguments(
    output_dir="./outputs",
    num_train_epochs=12,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    weight_decay=1e-4,
    warmup_steps=200,
    max_grad_norm=0.1,
    logging_steps=50,
    seed=42,
    fp16=True,
    dataloader_num_workers=4,
    lr_scheduler_type="linear"
):
    """
    Get TrainingArguments with default values optimized for RT-DETR.
    
    Args:
        output_dir: Directory to save model checkpoints
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device for training
        per_device_eval_batch_size: Batch size per device for evaluation
        gradient_accumulation_steps: Number of gradient accumulation steps
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        warmup_steps: Number of warmup steps
        max_grad_norm: Maximum gradient norm for clipping
        logging_steps: Log every N steps
        seed: Random seed
        fp16: Enable mixed precision training (faster with Tensor Cores)
        dataloader_num_workers: Number of worker processes for data loading
        
    Returns:
        TrainingArguments object
    """
    from transformers import TrainingArguments
    
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        
        # Batch size configuration
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        
        # Evaluation and saving
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        
        # Optimization
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        max_grad_norm=max_grad_norm,
        lr_scheduler_type=lr_scheduler_type,        # Performance optimization
        fp16=fp16,  # Enable mixed precision for faster training
        dataloader_pin_memory=True,
        dataloader_num_workers=dataloader_num_workers,  # Parallel data loading
        
        # Logging and monitoring
        logging_steps=logging_steps,
        logging_first_step=True,
        report_to="tensorboard",
        
        # Model selection
        load_best_model_at_end=True,
        metric_for_best_model="map",
        greater_is_better=True,
        
        # Other settings
        seed=seed,
        push_to_hub=False,
        remove_unused_columns=False,
        
        # Memory optimization
        optim="adamw_torch",
    )


class FreezeBackboneCallback(TrainerCallback):
    """Callback to freeze/unfreeze backbone at specific epochs"""
    
    def __init__(self, unfreeze_epoch=2):
        """
        Args:
            unfreeze_epoch: Epoch at which to unfreeze the backbone (0-indexed)
        """
        self.unfreeze_epoch = unfreeze_epoch
        self.unfrozen = False
    
    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        """Called at the beginning of each epoch"""
        current_epoch = int(state.epoch) if state.epoch is not None else 0
        
        if current_epoch < self.unfreeze_epoch and not self.unfrozen:
            # Freeze backbone
            frozen_count = 0
            for name, param in model.named_parameters():
                if 'backbone' in name or 'encoder' in name:
                    param.requires_grad = False
                    frozen_count += 1
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"\n{'='*60}")
            print(f"Epoch {current_epoch}: Backbone FROZEN")
            print(f"  Frozen {frozen_count} backbone/encoder layers")
            print(f"  Trainable: {trainable_params:,} / {total_params:,} "
                  f"({100 * trainable_params / total_params:.1f}%)")
            print(f"{'='*60}\n")
        
        elif current_epoch >= self.unfreeze_epoch and not self.unfrozen:
            # Unfreeze all
            for param in model.parameters():
                param.requires_grad = True
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"\n{'='*60}")
            print(f"Epoch {current_epoch}: Backbone UNFROZEN")
            print(f"  All layers now trainable")
            print(f"  Trainable: {trainable_params:,} / {total_params:,} "
                  f"({100 * trainable_params / total_params:.1f}%)")
            print(f"{'='*60}\n")
            
            self.unfrozen = True


def get_freeze_backbone_callback(unfreeze_epoch=2):
    """
    Get callback for freezing/unfreezing backbone.
    
    Args:
        unfreeze_epoch: Epoch at which to unfreeze (0-indexed)
    
    Returns:
        FreezeBackboneCallback: Configured callback
    """
    return FreezeBackboneCallback(unfreeze_epoch=unfreeze_epoch)


def get_early_stopping_callback(patience=3):
    """
    Get early stopping callback.
    
    Args:
        patience: Number of epochs to wait before stopping
    
    Returns:
        EarlyStoppingCallback: Configured callback
    """
    return EarlyStoppingCallback(
        early_stopping_patience=patience,
        early_stopping_threshold=0.0,
    )
