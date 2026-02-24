#!/usr/bin/env python3
"""
YOLO26 Training Script for AWS SageMaker.

This script trains a YOLO26 model using Ultralytics inside a SageMaker
PyTorch training container. It handles:
- Dataset splitting (if not pre-split)
- Training with configurable hyperparameters
- Optional W&B logging
- Saving artifacts to SageMaker output paths
"""

import argparse
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def get_sagemaker_paths() -> Dict[str, Path]:
    """
    Get SageMaker paths from environment variables with local fallbacks.

    Returns:
        Dict containing paths for training data, model output, and data output.
    """
    paths = {
        "training": Path(os.environ.get("SM_CHANNEL_TRAINING", "./test_data")),
        "model_dir": Path(os.environ.get("SM_MODEL_DIR", "./test_output/model")),
        "output_dir": Path(os.environ.get("SM_OUTPUT_DATA_DIR", "./test_output/output")),
    }

    # Create output directories
    paths["model_dir"].mkdir(parents=True, exist_ok=True)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    logger.info(f"Training data path: {paths['training']}")
    logger.info(f"Model output path: {paths['model_dir']}")
    logger.info(f"Data output path: {paths['output_dir']}")

    return paths


def get_num_gpus() -> int:
    """Get number of available GPUs."""
    num_gpus = int(os.environ.get("SM_NUM_GPUS", "0"))
    if num_gpus == 0:
        try:
            import torch
            num_gpus = torch.cuda.device_count()
        except ImportError:
            pass
    logger.info(f"Number of GPUs available: {num_gpus}")
    return num_gpus


def discover_dataset_structure(data_path: Path) -> Dict:
    """
    Discover the structure of the input dataset.

    Handles multiple formats:
    - Pre-split: images/train/, images/val/, images/test/ (all present)
    - Single split (FiftyOne default): images/val/ only - needs re-splitting
    - Unsplit: images/ contains image files directly

    Args:
        data_path: Path to the training data directory.

    Returns:
        Dict with dataset info including whether it's pre-split and class names.
    """
    images_dir = data_path / "images"
    labels_dir = data_path / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Check for split subdirectories
    train_images = images_dir / "train"
    val_images = images_dir / "val"
    test_images = images_dir / "test"

    # Determine dataset structure
    has_train = train_images.exists() and any(train_images.iterdir())
    has_val = val_images.exists() and any(val_images.iterdir())
    has_test = test_images.exists() and any(test_images.iterdir())

    # Check for images directly in images/ (flat structure)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    flat_images = [f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    if has_train and has_val:
        # Properly pre-split dataset
        structure = "presplit"
        source_images_dir = None
        source_labels_dir = None
    elif has_val and not has_train:
        # FiftyOne default export - only val split, needs re-splitting
        structure = "single_split"
        source_images_dir = val_images
        source_labels_dir = labels_dir / "val" if (labels_dir / "val").exists() else labels_dir
    elif flat_images:
        # Flat structure - images directly in images/
        structure = "flat"
        source_images_dir = images_dir
        source_labels_dir = labels_dir
    else:
        raise ValueError(f"Could not determine dataset structure in {data_path}")

    is_presplit = structure == "presplit"

    # Try to load existing data.yaml
    data_yaml_path = data_path / "data.yaml"
    dataset_yaml_path = data_path / "dataset.yaml"

    classes = []
    if data_yaml_path.exists():
        with open(data_yaml_path) as f:
            data_config = yaml.safe_load(f)
            names = data_config.get("names", [])
            # Handle both list and dict formats
            if isinstance(names, dict):
                classes = [names[i] for i in sorted(names.keys())]
            else:
                classes = names
    elif dataset_yaml_path.exists():
        with open(dataset_yaml_path) as f:
            data_config = yaml.safe_load(f)
            names = data_config.get("names", [])
            if isinstance(names, dict):
                classes = [names[i] for i in sorted(names.keys())]
            else:
                classes = names

    # If no classes found, try to infer from label files
    if not classes and labels_dir.exists():
        classes = infer_classes_from_labels(labels_dir, is_presplit)

    info = {
        "is_presplit": is_presplit,
        "structure": structure,
        "images_dir": images_dir,
        "labels_dir": labels_dir,
        "source_images_dir": source_images_dir,
        "source_labels_dir": source_labels_dir,
        "classes": classes,
        "existing_yaml": data_yaml_path if data_yaml_path.exists() else dataset_yaml_path if dataset_yaml_path.exists() else None,
    }

    logger.info(f"Dataset structure: {structure}")
    logger.info(f"Found {len(classes)} classes: {classes[:10]}{'...' if len(classes) > 10 else ''}")

    return info


def infer_classes_from_labels(labels_dir: Path, is_presplit: bool) -> List[str]:
    """
    Infer class indices from label files.

    Args:
        labels_dir: Path to labels directory.
        is_presplit: Whether the dataset is pre-split.

    Returns:
        List of class indices as strings (actual names unknown).
    """
    class_indices = set()

    if is_presplit:
        label_paths = list((labels_dir / "train").glob("*.txt")) + \
                      list((labels_dir / "val").glob("*.txt"))
    else:
        label_paths = list(labels_dir.glob("*.txt"))

    for label_path in label_paths[:100]:  # Sample first 100 files
        try:
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_indices.add(int(parts[0]))
        except (ValueError, IOError):
            continue

    # Return as list of indices (names unknown)
    max_idx = max(class_indices) if class_indices else 0
    return [f"class_{i}" for i in range(max_idx + 1)]


def split_dataset(
    data_path: Path,
    output_path: Path,
    source_images_dir: Optional[Path] = None,
    source_labels_dir: Optional[Path] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Path:
    """
    Split an unsplit dataset into train/val/test splits.

    Args:
        data_path: Path to dataset root.
        output_path: Path to write split dataset.
        source_images_dir: Directory containing source images (if different from data_path/images).
        source_labels_dir: Directory containing source labels (if different from data_path/labels).
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.
        seed: Random seed for reproducibility.

    Returns:
        Path to the split dataset directory.
    """
    logger.info(f"Splitting dataset with ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")

    random.seed(seed)

    # Use provided source directories or default to data_path subdirs
    images_dir = source_images_dir or (data_path / "images")
    labels_dir = source_labels_dir or (data_path / "labels")

    logger.info(f"Source images: {images_dir}")
    logger.info(f"Source labels: {labels_dir}")

    # Get all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_files = [f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    random.shuffle(image_files)

    # Calculate split indices
    n_total = len(image_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        "train": image_files[:n_train],
        "val": image_files[n_train:n_train + n_val],
        "test": image_files[n_train + n_val:],
    }

    logger.info(f"Split sizes: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # Create split directories and copy files
    for split_name, files in splits.items():
        split_images_dir = output_path / "images" / split_name
        split_labels_dir = output_path / "labels" / split_name
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)

        for img_file in files:
            # Copy image
            shutil.copy2(img_file, split_images_dir / img_file.name)

            # Copy corresponding label if it exists
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy2(label_file, split_labels_dir / label_file.name)

    return output_path


def create_data_yaml(
    output_path: Path,
    classes: List[str],
    is_presplit: bool = True,
) -> Path:
    """
    Create data.yaml file for YOLO training.

    Args:
        output_path: Path to dataset directory.
        classes: List of class names.
        is_presplit: Whether the dataset is pre-split.

    Returns:
        Path to created data.yaml file.
    """
    data_yaml_path = output_path / "data.yaml"

    data_config = {
        "path": str(output_path.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(classes)},
    }

    with open(data_yaml_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)

    logger.info(f"Created data.yaml at {data_yaml_path}")
    return data_yaml_path


def setup_wandb(args: argparse.Namespace) -> bool:
    """
    Initialize Weights & Biases for Ultralytics integration.

    We pre-initialize W&B with the correct project name before training.
    Ultralytics will detect the existing run and use it for logging.

    Args:
        args: Parsed command line arguments.

    Returns:
        True if W&B is initialized, False otherwise.
    """
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        logger.info("WANDB_API_KEY not set, W&B logging disabled")
        return False

    try:
        import wandb
        from ultralytics import settings

        # Enable W&B in Ultralytics settings
        settings.update(wandb=True)
        logger.info("Enabled W&B in Ultralytics settings")

        # Get project and run name
        wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT", "yolo-sagemaker")
        wandb_name = args.wandb_name or f"yolo26-{args.epochs}ep"
        wandb_entity = os.environ.get("WANDB_ENTITY")

        # Pre-initialize W&B with correct project
        # Ultralytics will detect this run and use it
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_name,
            config={
                "model": args.model,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "imgsz": args.imgsz,
                "lr0": args.lr0,
                "patience": args.patience,
            },
            resume="allow",
        )

        logger.info(f"W&B initialized: project={wandb_project}, name={wandb_name}")
        return True
    except ImportError:
        logger.warning("wandb not installed, W&B logging disabled")
        return False
    except Exception as e:
        logger.warning(f"Failed to initialize W&B: {e}")
        return False


def train_model(
    data_yaml: Path,
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[Path, Dict]:
    """
    Train YOLO model using Ultralytics.

    Args:
        data_yaml: Path to data.yaml file.
        args: Parsed command line arguments.
        output_dir: Directory for training outputs.

    Returns:
        Tuple of (path to best weights, training metrics dict).
    """
    from ultralytics import YOLO

    logger.info(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Build training arguments
    train_args = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "batch": args.batch_size,
        "imgsz": args.imgsz,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "patience": args.patience,
        "project": str(output_dir / "train"),
        "name": "yolo_training",
        "exist_ok": True,
        "verbose": True,
    }

    # Add augmentation args if specified
    aug_args = {
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "degrees": args.degrees,
        "translate": args.translate,
        "scale": args.scale,
        "shear": args.shear,
        "flipud": args.flipud,
        "fliplr": args.fliplr,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "copy_paste": args.copy_paste,
    }

    # Only add non-None augmentation args
    for key, value in aug_args.items():
        if value is not None:
            train_args[key] = value

    # Handle multi-GPU
    num_gpus = get_num_gpus()
    if num_gpus > 1:
        train_args["device"] = list(range(num_gpus))
        logger.info(f"Using {num_gpus} GPUs for training")
    elif num_gpus == 1:
        train_args["device"] = 0
        logger.info("Using single GPU for training")
    else:
        train_args["device"] = "cpu"
        logger.info("Using CPU for training")

    logger.info(f"Starting training with args: {train_args}")
    results = model.train(**train_args)

    # Get best weights path
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = Path(results.save_dir) / "weights" / "last.pt"

    # Extract metrics
    metrics = {}
    if hasattr(results, "results_dict"):
        metrics = results.results_dict

    logger.info(f"Training complete. Best weights: {best_weights}")
    return best_weights, metrics


def evaluate_model(
    model_path: Path,
    data_yaml: Path,
    output_dir: Path,
) -> Dict:
    """
    Evaluate trained model on test split.

    Args:
        model_path: Path to trained model weights.
        data_yaml: Path to data.yaml file.
        output_dir: Directory for evaluation outputs.

    Returns:
        Dict of evaluation metrics.
    """
    from ultralytics import YOLO

    logger.info(f"Evaluating model on test split")
    model = YOLO(str(model_path))

    results = model.val(
        data=str(data_yaml),
        split="test",
        project=str(output_dir / "test"),
        name="evaluation",
        exist_ok=True,
    )

    metrics = {
        "mAP50": float(results.box.map50) if hasattr(results.box, "map50") else None,
        "mAP50-95": float(results.box.map) if hasattr(results.box, "map") else None,
        "precision": float(results.box.mp) if hasattr(results.box, "mp") else None,
        "recall": float(results.box.mr) if hasattr(results.box, "mr") else None,
    }

    logger.info(f"Evaluation metrics: {metrics}")
    return metrics


def save_artifacts(
    best_weights: Path,
    model_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    train_metrics: Dict,
    eval_metrics: Dict,
):
    """
    Save model artifacts to SageMaker output directories.

    Args:
        best_weights: Path to best model weights.
        model_dir: SageMaker model output directory.
        output_dir: SageMaker data output directory.
        args: Training arguments.
        train_metrics: Training metrics.
        eval_metrics: Evaluation metrics.
    """
    # Copy best weights to model directory
    model_path = model_dir / "best.pt"
    shutil.copy2(best_weights, model_path)
    logger.info(f"Copied best weights to {model_path}")

    # Also save as model.pt for easy discovery
    shutil.copy2(best_weights, model_dir / "model.pt")

    # Save training configuration
    config = {
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "imgsz": args.imgsz,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "patience": args.patience,
        "split_train": args.split_train,
        "split_val": args.split_val,
        "split_test": args.split_test,
        "split_seed": args.split_seed,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
    }

    config_path = model_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved training config to {config_path}")

    # Save metrics for SageMaker
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"train": train_metrics, "eval": eval_metrics}, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YOLO26 Training Script for SageMaker")

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26n.pt",
        help="Base model (yolo26n.pt, yolo26s.pt, yolo26m.pt, yolo26l.pt, yolo26x.pt)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="detect",
        choices=["detect", "segment", "classify", "pose"],
        help="Task type",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate factor")
    parser.add_argument("--momentum", type=float, default=0.937, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("--warmup-epochs", type=float, default=3.0, help="Warmup epochs")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")

    # Data augmentation
    parser.add_argument("--hsv-h", type=float, default=None, help="HSV-Hue augmentation")
    parser.add_argument("--hsv-s", type=float, default=None, help="HSV-Saturation augmentation")
    parser.add_argument("--hsv-v", type=float, default=None, help="HSV-Value augmentation")
    parser.add_argument("--degrees", type=float, default=None, help="Rotation augmentation")
    parser.add_argument("--translate", type=float, default=None, help="Translation augmentation")
    parser.add_argument("--scale", type=float, default=None, help="Scale augmentation")
    parser.add_argument("--shear", type=float, default=None, help="Shear augmentation")
    parser.add_argument("--flipud", type=float, default=None, help="Flip up-down probability")
    parser.add_argument("--fliplr", type=float, default=None, help="Flip left-right probability")
    parser.add_argument("--mosaic", type=float, default=None, help="Mosaic augmentation probability")
    parser.add_argument("--mixup", type=float, default=None, help="Mixup augmentation probability")
    parser.add_argument("--copy-paste", type=float, default=None, help="Copy-paste augmentation probability")

    # Dataset splitting
    parser.add_argument(
        "--pre-split",
        type=str,
        default="false",
        help="Whether data is pre-split into train/val/test (true/false)",
    )
    parser.add_argument("--split-train", type=float, default=0.8, help="Training split ratio")
    parser.add_argument("--split-val", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--split-test", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed for splitting")

    # W&B configuration
    parser.add_argument("--wandb-project", type=str, default=None, help="W&B project name")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb-tags", type=str, default=None, help="Comma-separated W&B tags")

    return parser.parse_args()


def main():
    """Main training entrypoint."""
    args = parse_args()
    logger.info(f"Starting YOLO training with args: {args}")

    # Get SageMaker paths
    paths = get_sagemaker_paths()

    # Setup W&B if configured
    wandb_enabled = setup_wandb(args)

    # Discover dataset structure
    dataset_info = discover_dataset_structure(paths["training"])

    # Prepare dataset
    force_presplit = args.pre_split.lower() == "true"
    if force_presplit or dataset_info["is_presplit"]:
        # Use existing split
        data_path = paths["training"]
        logger.info("Using pre-split dataset" + (" (forced via --pre-split)" if force_presplit else ""))
    else:
        # Split the dataset (handles both flat and single-split structures)
        data_path = paths["output_dir"] / "dataset"
        split_dataset(
            paths["training"],
            data_path,
            source_images_dir=dataset_info.get("source_images_dir"),
            source_labels_dir=dataset_info.get("source_labels_dir"),
            train_ratio=args.split_train,
            val_ratio=args.split_val,
            test_ratio=args.split_test,
            seed=args.split_seed,
        )

    # Create data.yaml
    data_yaml = create_data_yaml(
        data_path,
        dataset_info["classes"],
        is_presplit=True,  # After splitting, it's always pre-split
    )

    # Train model
    best_weights, train_metrics = train_model(data_yaml, args, paths["output_dir"])

    # Evaluate on test split
    eval_metrics = evaluate_model(best_weights, data_yaml, paths["output_dir"])

    # Save artifacts
    save_artifacts(
        best_weights,
        paths["model_dir"],
        paths["output_dir"],
        args,
        train_metrics,
        eval_metrics,
    )

    logger.info("Training complete!")

    if wandb_enabled:
        try:
            import wandb
            # Log final evaluation metrics
            wandb.log({
                "eval/mAP50": eval_metrics.get("mAP50"),
                "eval/mAP50-95": eval_metrics.get("mAP50-95"),
                "eval/precision": eval_metrics.get("precision"),
                "eval/recall": eval_metrics.get("recall"),
            })
            wandb.finish()
            logger.info("W&B run completed - check your W&B dashboard for training charts")
        except Exception as e:
            logger.warning(f"Error finishing W&B run: {e}")


if __name__ == "__main__":
    main()
