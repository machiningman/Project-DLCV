"""Data utilities for loading and augmenting datasets"""

import supervision as sv
import numpy as np
import albumentations as A
import torch
import cv2
from torch.utils.data import Dataset
from transformers import RTDetrImageProcessor


def load_datasets(coco_dir, coco_rain_dir, coco_ratio=0.8, rain_ratio=0.2, seed=42):
    """
    Load and combine COCO and COCO_rain datasets.
    
    Args:
        coco_dir: Path to COCO dataset directory
        coco_rain_dir: Path to COCO_rain dataset directory
        coco_ratio: Proportion of COCO images (default: 0.8)
        rain_ratio: Proportion of rain images (default: 0.2)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Load COCO dataset
    ds_train_coco = sv.DetectionDataset.from_coco(
        images_directory_path=f"{coco_dir}/train2017",
        annotations_path=f"{coco_dir}/annotations/instances_train2017.json",
    )
    
    ds_valid_coco = sv.DetectionDataset.from_coco(
        images_directory_path=f"{coco_dir}/val2017",
        annotations_path=f"{coco_dir}/annotations/instances_val2017.json",
    )
    
    # Load COCO_rain dataset
    ds_train_rain = sv.DetectionDataset.from_coco(
        images_directory_path=f"{coco_rain_dir}/train2017",
        annotations_path=f"{coco_rain_dir}/annotations/instances_train2017.json",
    )
    
    ds_valid_rain = sv.DetectionDataset.from_coco(
        images_directory_path=f"{coco_rain_dir}/val2017",
        annotations_path=f"{coco_rain_dir}/annotations/instances_val2017.json",
    )
    
    # Set target sizes
    target_train_size = len(ds_train_coco)
    target_val_size = len(ds_valid_coco)
    
    # Calculate sample sizes
    coco_train_size = int(target_train_size * coco_ratio)
    rain_train_size = int(target_train_size * rain_ratio)
    coco_val_size = int(target_val_size * coco_ratio)
    rain_val_size = int(target_val_size * rain_ratio)
    
    print(f"Target combined training size: {coco_train_size + rain_train_size}")
    print(f"  - COCO: {coco_train_size}")
    print(f"  - COCO_rain: {rain_train_size}")
    print(f"Target combined validation size: {coco_val_size + rain_val_size}")
    print(f"  - COCO: {coco_val_size}")
    print(f"  - COCO_rain: {rain_val_size}")
    
    # Create random indices for sampling
    np.random.seed(seed)
    coco_train_indices = np.random.choice(len(ds_train_coco), coco_train_size, replace=False)
    rain_train_indices = np.random.choice(len(ds_train_rain), rain_train_size, replace=False)
    coco_val_indices = np.random.choice(len(ds_valid_coco), coco_val_size, replace=False)
    rain_val_indices = np.random.choice(len(ds_valid_rain), rain_val_size, replace=False)
    
    # Subset datasets
    coco_train_paths = [ds_train_coco.image_paths[i] for i in coco_train_indices]
    rain_train_paths = [ds_train_rain.image_paths[i] for i in rain_train_indices]
    coco_val_paths = [ds_valid_coco.image_paths[i] for i in coco_val_indices]
    rain_val_paths = [ds_valid_rain.image_paths[i] for i in rain_val_indices]
    
    # Merge datasets
    train_combined_paths = coco_train_paths + rain_train_paths
    train_combined_annotations = {}
    train_combined_annotations.update({path: ds_train_coco.annotations[path] for path in coco_train_paths})
    train_combined_annotations.update({path: ds_train_rain.annotations[path] for path in rain_train_paths})
    
    val_combined_paths = coco_val_paths + rain_val_paths
    val_combined_annotations = {}
    val_combined_annotations.update({path: ds_valid_coco.annotations[path] for path in coco_val_paths})
    val_combined_annotations.update({path: ds_valid_rain.annotations[path] for path in rain_val_paths})
    
    # Create merged datasets
    ds_train = sv.DetectionDataset(
        classes=ds_train_coco.classes,
        images=train_combined_paths,
        annotations=train_combined_annotations
    )
    
    ds_valid = sv.DetectionDataset(
        classes=ds_valid_coco.classes,
        images=val_combined_paths,
        annotations=val_combined_annotations
    )
    
    print(f"✓ Combined datasets created:")
    print(f"  Training: {len(ds_train)} images ({len(coco_train_paths)} COCO + {len(rain_train_paths)} COCO_rain)")
    print(f"  Validation: {len(ds_valid)} images ({len(coco_val_paths)} COCO + {len(rain_val_paths)} COCO_rain)")
    
    return ds_train, ds_valid


def get_augmentation_transforms(is_train=True):
    """
    Get augmentation transforms for training or validation.
    
    Args:
        is_train: If True, returns training augmentations, else validation augmentations
        
    Returns:
        Albumentations Compose object
    """
    if is_train:
        return A.Compose(
            [
                A.Perspective(p=0.1, scale=(0.05, 0.1)),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.1),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["category"],
                min_area=25,
                min_visibility=0.3,
                clip=True
            ),
        )
    else:
        return A.Compose(
            [A.NoOp()],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category"], clip=True, min_area=1),
        )


def get_domain_augmentation_transforms():
    """Return (clean_aug, rain_aug) where rain_aug is lighter to avoid compounding degradations."""
    clean = A.Compose(
        [
            A.Perspective(p=0.1, scale=(0.05, 0.1)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.1),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["category"],
            min_area=25,
            min_visibility=0.3,
            clip=True,
        ),
    )
    rain = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),  # lighter
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["category"],
            min_area=25,
            min_visibility=0.2,
            clip=True,
        ),
    )
    return clean, rain


class AugmentedDetectionDataset(Dataset):
    """PyTorch Dataset for object detection with domain-aware augmentations"""

    def __init__(self, dataset, processor, transform=None, is_train=True, transform_clean=None, transform_rain=None):
        self.dataset = dataset
        self.processor = processor
        self.transform = transform  # legacy single-transform path
        self.transform_clean = transform_clean
        self.transform_rain = transform_rain
        self.is_train = is_train
        # Attempt to cache image paths to support domain decisions
        self._paths = None
        if hasattr(dataset, "image_paths"):
            self._paths = list(dataset.image_paths)
        elif hasattr(dataset, "images"):
            self._paths = list(dataset.images)

    def get_path(self, idx):
        if self._paths is None:
            return None
        return self._paths[idx]

    def _is_rain(self, idx):
        p = self.get_path(idx)
        if p is None:
            return False
        p = str(p).lower()
        return ("coco_rain" in p) or ("rain" in p)

    @staticmethod
    def annotations_as_coco(image_id, categories, areas, boxes):
        """Convert annotations to COCO format"""
        annotations = []
        for category, area, box in zip(categories, areas, boxes):
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            
            if width <= 0 or height <= 0:
                continue
                
            annotations.append({
                "image_id": image_id,
                "category_id": int(category),
                "iscrowd": 0,
                "area": area if area else width * height,
                "bbox": [x_min, y_min, width, height],
            })
        return {"image_id": image_id, "annotations": annotations}
    
    @staticmethod
    def validate_bbox(box, min_size=10):
        """Validate bounding box dimensions"""
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        if width < min_size or height < min_size:
            return False
        if x1 < 0 or y1 < 0:
            return False
        if x2 <= x1 or y2 <= y1:
            return False
        if np.isnan(box).any() or np.isinf(box).any():
            return False
        
        return True

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                return self._get_item_internal(idx)
            except (ValueError, AssertionError) as e:
                if attempt < max_retries - 1:
                    idx = (idx + 1) % len(self.dataset)
                    continue
                else:
                    print(f"Warning: Failed to load sample after {max_retries} attempts. Creating dummy sample.")
                    return self._create_dummy_sample()
    
    def _create_dummy_sample(self):
        """Create a valid dummy sample when real data fails"""
        dummy_image = np.ones((640, 640, 3), dtype=np.uint8) * 128
        dummy_box = [[256, 256, 384, 384]]
        dummy_category = [0]
        dummy_area = [128 * 128]
        
        formatted_annotations = self.annotations_as_coco(0, dummy_category, dummy_area, dummy_box)
        result = self.processor(images=dummy_image, annotations=formatted_annotations, return_tensors="pt")
        
        result["pixel_values"] = result["pixel_values"].squeeze(0)
        if "pixel_mask" in result:
            result["pixel_mask"] = result["pixel_mask"].squeeze(0)
        result["labels"] = result["labels"][0]
        result["labels"]["orig_size"] = torch.tensor([640, 640], dtype=torch.int64)
        
        return result

    def _get_item_internal(self, idx):
        """Internal method to get item with full processing"""
        image_id, image, annotations = self.dataset[idx]
        if image is None:
            raise ValueError(f"Image not loaded for index {idx} (image_id: {image_id}).")
        
        # Ensure image is RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = np.array(image)
        else:
            image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        
        # PRE-FILTER: Remove invalid boxes BEFORE augmentation
        valid_indices = []
        for i, box in enumerate(annotations.xyxy):
            if self.validate_bbox(box, min_size=10):
                valid_indices.append(i)
        
        if len(valid_indices) == 0:
            h, w = image.shape[:2]
            valid_boxes = np.array([[w*0.4, h*0.4, w*0.6, h*0.6]])
            valid_categories = np.array([0])
        else:
            valid_boxes = annotations.xyxy[valid_indices]
            valid_categories = annotations.class_id[valid_indices]
        
        # Choose transform based on domain (rain vs clean)
        use_transform = self.transform
        if self.transform_clean is not None and self.transform_rain is not None:
            use_transform = self.transform_rain if self._is_rain(idx) else self.transform_clean

        # Apply transformations
        transformed = use_transform(
            image=image,
            bboxes=valid_boxes,
            category=valid_categories
        )
        
        image = transformed["image"]
        boxes = transformed["bboxes"]
        categories = transformed["category"]
        
        # POST-FILTER: Validate boxes after transformation
        final_boxes = []
        final_categories = []
        
        h, w = image.shape[:2]
        for box, cat in zip(boxes, categories):
            x1, y1, x2, y2 = box
            
            # Clip to image boundaries
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            # Validate after clipping
            if self.validate_bbox([x1, y1, x2, y2], min_size=10):
                final_boxes.append([x1, y1, x2, y2])
                final_categories.append(cat)
        
        # Ensure we have at least one box
        if len(final_boxes) == 0:
            h, w = image.shape[:2]
            final_boxes = [[w*0.4, h*0.4, w*0.6, h*0.6]]
            final_categories = [0]
        
        boxes = final_boxes
        categories = final_categories
        
        # Calculate areas
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
        areas = [max(area, 100.0) for area in areas]
        
        # Format annotations for processor
        formatted_annotations = self.annotations_as_coco(idx, categories, areas, boxes)
        
        # Process with HuggingFace processor
        result = self.processor(images=image, annotations=formatted_annotations, return_tensors="pt")
        
        # Squeeze batch dimensions
        result["pixel_values"] = result["pixel_values"].squeeze(0)
        if "pixel_mask" in result:
            result["pixel_mask"] = result["pixel_mask"].squeeze(0)
        result["labels"] = result["labels"][0]
        
        # Add original size for post-processing
        h, w = image.shape[:2]
        result["labels"]["orig_size"] = torch.tensor([h, w], dtype=torch.int64)
        
        # Final validation
        if torch.isnan(result["pixel_values"]).any():
            raise ValueError(f"NaN detected in pixel_values at index {idx}")
        if "boxes" in result["labels"] and len(result["labels"]["boxes"]) > 0:
            if torch.isnan(result["labels"]["boxes"]).any():
                raise ValueError(f"NaN detected in boxes at index {idx}")
        
        return result


def create_detection_datasets(ds_train, ds_valid, processor, percent_dataset=100):
    """
    Create augmented detection datasets for training and validation.
    
    Args:
        ds_train: Training supervision dataset
        ds_valid: Validation supervision dataset
        processor: RTDetrImageProcessor
        percent_dataset: Percentage of dataset to use (default: 100)
        
    Returns:
        Tuple of (train_dataset, valid_dataset)
    """
    augmentation_valid = get_augmentation_transforms(is_train=False)
    # Domain-aware training augmentations
    aug_clean, aug_rain = get_domain_augmentation_transforms()
    
    split_ratio = percent_dataset / 100
    train_subset, _ = ds_train.split(split_ratio=split_ratio)
    valid_subset, _ = ds_valid.split(split_ratio=split_ratio)
    
    train_dataset = AugmentedDetectionDataset(
        train_subset,
        processor,
        transform=None,
        is_train=True,
        transform_clean=aug_clean,
        transform_rain=aug_rain,
    )
    valid_dataset = AugmentedDetectionDataset(valid_subset, processor, augmentation_valid, is_train=False)
    
    print(f"✓ Datasets created")
    print(f"  Training samples: {len(train_dataset):,}")
    print(f"  Validation samples: {len(valid_dataset):,}")
    
    return train_dataset, valid_dataset


def split_by_domain(ds):
    """Split a supervision DetectionDataset into (clean_ds, rain_ds) by path string."""
    clean_paths, rain_paths = [], []
    clean_ann, rain_ann = {}, {}
    paths = getattr(ds, "image_paths", None) or getattr(ds, "images", None)
    if paths is None:
        return ds, ds  # fallback
    for p in paths:
        p_str = str(p).lower()
        if ("coco_rain" in p_str) or ("rain" in p_str):
            rain_paths.append(p)
            rain_ann[p] = ds.annotations[p]
        else:
            clean_paths.append(p)
            clean_ann[p] = ds.annotations[p]

    classes = getattr(ds, "classes", None)
    clean_ds = sv.DetectionDataset(classes=classes, images=clean_paths, annotations=clean_ann)
    rain_ds = sv.DetectionDataset(classes=classes, images=rain_paths, annotations=rain_ann)
    return clean_ds, rain_ds


def check_split_leakage(ds_train, ds_valid):
    """Print potential leakage count by comparing image basenames (without extension)."""
    def _stems(ds):
        paths = getattr(ds, "image_paths", None) or getattr(ds, "images", None)
        if not paths:
            return set()
        import os
        return {os.path.splitext(os.path.basename(str(p)))[0] for p in paths}

    tr, va = _stems(ds_train), _stems(ds_valid)
    overlap = tr & va
    print(f"Leakage check: {len(overlap)} overlapping image stems between train and val")


def collate_fn(batch):
    """Collate function for DataLoader"""
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    return data
