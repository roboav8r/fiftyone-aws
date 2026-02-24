"""
YOLO26 SageMaker Training Plugin for FiftyOne.

This plugin provides operators and a panel to train YOLO26 models on AWS
SageMaker with full configuration options for classified AWS networks
(GovCloud, IC regions).

Features:
- Custom ECR image URI support (required for classified networks)
- VPC configuration (subnets, security groups)
- Full hyperparameter control
- Optional W&B integration
- Job monitoring and artifact download
- Python panel UI with Train/Monitor/Apply tabs
- Model inference application
"""

import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types

from fiftyone.operators.panel import Panel, PanelConfig

from .sagemaker_utils import (
    SageMakerClient,
    SageMakerConfig,
    TrainingJobConfig,
)

logger = logging.getLogger(__name__)

# Plugin directory for config file lookup
PLUGIN_DIR = Path(__file__).parent
CONFIG_FILE = PLUGIN_DIR / "default_config.yaml"

# Secret key names (declared in fiftyone.yml under secrets:)
# Resolved via FiftyOne Secrets Manager (Teams) or environment variables (OSS)
SECRET_AWS_ACCESS_KEY_ID = "AWS_ACCESS_KEY_ID"
SECRET_AWS_SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY"
SECRET_WANDB_API_KEY = "WANDB_API_KEY"


def _load_default_config() -> Dict[str, Any]:
    """Load default configuration from YAML file.

    Returns:
        Configuration dictionary, or empty dict if file not found.
    """
    if not CONFIG_FILE.exists():
        logger.debug(f"No config file found at {CONFIG_FILE}")
        return {}

    try:
        with open(CONFIG_FILE, "r") as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"Loaded default config from {CONFIG_FILE}")
        return config
    except Exception as e:
        logger.warning(f"Failed to load config file: {e}")
        return {}


def _resolve_aws_credentials(ctx) -> Dict[str, Optional[str]]:
    """Resolve AWS credentials from FiftyOne secrets.

    Uses Secrets Manager (Teams) or environment variables (open-source).

    Args:
        ctx: Operator execution context.

    Returns:
        Dict with aws_access_key_id and aws_secret_access_key (may be None).
    """
    try:
        secrets = ctx.secrets
        return {
            "aws_access_key_id": secrets.get(SECRET_AWS_ACCESS_KEY_ID) or None,
            "aws_secret_access_key": secrets.get(SECRET_AWS_SECRET_ACCESS_KEY) or None,
        }
    except Exception:
        return {"aws_access_key_id": None, "aws_secret_access_key": None}


def _resolve_wandb_api_key(ctx) -> Optional[str]:
    """Resolve W&B API key from FiftyOne secrets.

    Args:
        ctx: Operator execution context.

    Returns:
        W&B API key string or None.
    """
    try:
        return ctx.secrets.get(SECRET_WANDB_API_KEY) or None
    except Exception:
        return None



def _get_config_value(config: Dict[str, Any], *keys, default=None):
    """Get a nested config value safely.

    Args:
        config: Configuration dictionary.
        *keys: Nested keys to traverse.
        default: Default value if not found.

    Returns:
        Config value or default.
    """
    value = config
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return default
        if value is None:
            return default
    # Return default for empty strings
    if value == "":
        return default
    return value


def _export_dataset_to_yolo(
    dataset,
    export_dir: Path,
    label_field: str = "ground_truth",
    classes: Optional[List[str]] = None,
    split: str = "val",
) -> Path:
    """Export FiftyOne dataset or view to YOLO format.

    Args:
        dataset: FiftyOne dataset or DatasetView to export.
        export_dir: Directory to export to.
        label_field: Field containing detections.
        classes: List of class names (auto-detected if None).
        split: Split name for YOLO export directory structure.

    Returns:
        Path to exported dataset.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect classes if not provided
    if classes is None:
        det_label_field = f"{label_field}.detections.label"
        classes = dataset.distinct(det_label_field)
        logger.info(f"Auto-detected {len(classes)} classes: {classes[:10]}...")

    # Export to YOLO format
    dataset.export(
        export_dir=str(export_dir),
        dataset_type=fo.types.YOLOv5Dataset,
        label_field=label_field,
        classes=classes,
        split=split,
    )

    logger.info(f"Exported {len(dataset)} samples to {export_dir}")
    return export_dir


def _parse_tags(tags_str: Optional[str]) -> Dict[str, str]:
    """Parse comma-separated key=value tags into dict."""
    if not tags_str:
        return {}
    tags = {}
    for pair in tags_str.split(","):
        pair = pair.strip()
        if "=" in pair:
            key, value = pair.split("=", 1)
            tags[key.strip()] = value.strip()
    return tags


def _parse_list(list_str: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated string into list."""
    if not list_str:
        return None
    return [s.strip() for s in list_str.split(",") if s.strip()]


def _parse_json_dict(json_str: Optional[str]) -> Dict[str, Any]:
    """Parse JSON string into dict."""
    if not json_str:
        return {}
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON: {json_str}")
        return {}


def _build_sagemaker_config_from_params(
    params: Dict[str, Any],
    ctx=None,
) -> SageMakerConfig:
    """Build a SageMakerConfig from operator params and secrets.

    Extracts common AWS/SageMaker fields from params dict.
    AWS credentials are resolved from ctx.secrets (never from params).

    Args:
        params: Operator parameters dict.
        ctx: Operator execution context (for resolving secrets).

    Returns:
        SageMakerConfig instance.
    """
    creds = _resolve_aws_credentials(ctx) if ctx else {}
    return SageMakerConfig(
        role=params.get("role", ""),
        bucket=params.get("bucket", ""),
        image_uri=params.get("image_uri", ""),
        region=params.get("region", "us-east-1"),
        profile=params.get("profile") or None,
        aws_access_key_id=creds.get("aws_access_key_id"),
        aws_secret_access_key=creds.get("aws_secret_access_key"),
    )


class ApplySplitTags(foo.Operator):
    """Operator to apply random train/val/test split tags to samples."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="apply_split_tags",
            label="Apply Split Tags",
            description="Apply random train/val/test tags to samples in the current view",
            unlisted=True,
            allow_immediate_execution=True,
        )

    def resolve_input(self, ctx):
        """Define the input form for the operator."""
        inputs = types.Object()
        inputs.float("split_train", label="Train Split", required=True)
        inputs.float("split_val", label="Validation Split", required=True)
        inputs.float("split_test", label="Test Split", required=True)
        return types.Property(inputs)

    def execute(self, ctx):
        """Apply random split tags to samples in the current view.

        Clears existing train/val/test tags from the current view, then
        uses fiftyone.utils.random.random_split() to apply new tags.

        Returns:
            dict: Result with updated tag counts and sample count.
        """
        import fiftyone.utils.random as four

        params = ctx.params
        split_train = params.get("split_train", 0.8)
        split_val = params.get("split_val", 0.1)
        split_test = params.get("split_test", 0.1)

        view = ctx.view
        if view is None:
            view = ctx.dataset

        if view is None:
            return {"status": "error", "message": "No dataset or view available"}

        # Clear existing train/val/test tags from the current view
        for tag in ["train", "val", "test"]:
            tagged = view.match_tags(tag)
            if len(tagged) > 0:
                tagged.untag_samples(tag)

        # Build split fractions dict
        splits = {"train": split_train, "val": split_val}
        if split_test > 0:
            splits["test"] = split_test

        # Apply random split tags
        four.random_split(view, splits)

        # Return updated tag counts
        tag_counts = view.count_sample_tags()

        return {
            "status": "success",
            "tag_counts": tag_counts,
            "sample_count": len(view),
        }


class LaunchSageMakerTraining(foo.Operator):
    """Operator to launch YOLO26 training jobs on AWS SageMaker."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="launch_sagemaker_training",
            label="Launch SageMaker Training",
            description="Train a YOLO26 model on AWS SageMaker",
            unlisted=False,
            allow_immediate_execution=False,
            allow_delegated_execution=True,
            default_choice_to_delegated=True,
        )

    def resolve_input(self, ctx):
        """Define the input form for the operator."""
        inputs = types.Object()

        # Load default config
        cfg = _load_default_config()

        # === Required SageMaker Configuration ===
        inputs.str(
            "image_uri",
            label="ECR Image URI",
            description="Docker image URI from ECR (required for classified networks)",
            default=_get_config_value(cfg, "sagemaker", "image_uri"),
            required=True,
        )
        inputs.str(
            "role",
            label="IAM Role ARN",
            description="SageMaker execution role ARN",
            default=_get_config_value(cfg, "sagemaker", "role"),
            required=True,
        )
        inputs.str(
            "bucket",
            label="S3 Bucket",
            description="S3 bucket for training data and artifacts",
            default=_get_config_value(cfg, "sagemaker", "bucket"),
            required=True,
        )
        inputs.int(
            "instance_count",
            label="Instance Count",
            description="Number of training instances",
            default=_get_config_value(cfg, "sagemaker", "instance_count", default=1),
            required=True,
        )
        inputs.str(
            "instance_type",
            label="Instance Type",
            description="EC2 instance type (e.g., ml.g4dn.xlarge, ml.p3.2xlarge)",
            default=_get_config_value(cfg, "sagemaker", "instance_type", default="ml.g4dn.xlarge"),
            required=True,
        )
        inputs.str(
            "subnets",
            label="VPC Subnets",
            description="Comma-separated VPC subnet IDs (optional, required for classified networks)",
            default=_get_config_value(cfg, "sagemaker", "subnets"),
            required=False,
        )
        inputs.str(
            "security_group_ids",
            label="Security Groups",
            description="Comma-separated VPC security group IDs (optional)",
            default=_get_config_value(cfg, "sagemaker", "security_group_ids"),
            required=False,
        )
        inputs.str(
            "tags",
            label="Resource Tags",
            description="Comma-separated key=value pairs (e.g., project=demo,team=ml)",
            default=_get_config_value(cfg, "sagemaker", "tags"),
            required=False,
        )

        # === Optional SageMaker Configuration ===
        inputs.str(
            "output_path",
            label="Output S3 Path",
            description="S3 path for model outputs (default: s3://{bucket}/output)",
            default=_get_config_value(cfg, "sagemaker", "output_path"),
            required=False,
        )
        inputs.str(
            "code_location",
            label="Code S3 Path",
            description="S3 path for uploaded code (default: s3://{bucket}/code)",
            default=_get_config_value(cfg, "sagemaker", "code_location"),
            required=False,
        )
        inputs.str(
            "base_job_name",
            label="Base Job Name",
            description="Prefix for training job names",
            default=_get_config_value(cfg, "sagemaker", "base_job_name", default="yolo26-training"),
            required=False,
        )
        inputs.bool(
            "disable_profiler",
            label="Disable Profiler",
            description="Disable SageMaker Debugger profiler",
            default=_get_config_value(cfg, "sagemaker", "disable_profiler", default=True),
        )
        inputs.str(
            "dependencies",
            label="Dependencies",
            description="Comma-separated additional pip dependencies",
            required=False,
        )
        inputs.bool(
            "enable_sagemaker_metrics",
            label="Enable SageMaker Metrics",
            description="Enable CloudWatch metrics collection",
            default=_get_config_value(cfg, "sagemaker", "enable_sagemaker_metrics", default=False),
        )

        # === Dataset Configuration ===
        inputs.str(
            "label_field",
            label="Label Field",
            description="FiftyOne field containing detection labels",
            default=_get_config_value(cfg, "dataset", "label_field", default="ground_truth"),
            required=True,
        )

        # === Split Mode ===
        inputs.str(
            "split_mode",
            label="Split Mode",
            description="How to split data: 'ratio' for random split, 'saved_views' for pre-defined views",
            default="ratio",
            required=False,
        )
        inputs.str(
            "train_view",
            label="Train View",
            description="Name of saved view for training data (saved_views mode)",
            required=False,
        )
        inputs.str(
            "val_view",
            label="Validation View",
            description="Name of saved view for validation data (saved_views mode)",
            required=False,
        )
        inputs.str(
            "test_view",
            label="Test View",
            description="Name of saved view for test data (saved_views mode, optional)",
            required=False,
        )

        # === Tag-based Split Configuration ===
        inputs.str(
            "train_tag",
            label="Train Tag",
            description="Sample tag for training data (tags mode)",
            required=True,
        )
        inputs.str(
            "val_tag",
            label="Validation Tag",
            description="Sample tag for validation data (tags mode)",
            required=True,
        )
        inputs.str(
            "test_tag",
            label="Test Tag",
            description="Sample tag for test data (tags mode, optional)",
            required=False,
        )

        # === Training Hyperparameters ===
        inputs.str(
            "model",
            label="YOLO Model",
            description="Base model (yolo26n.pt, yolo26s.pt, yolo26m.pt, yolo26l.pt, yolo26x.pt)",
            default=_get_config_value(cfg, "training", "model", default="yolo26n.pt"),
            required=True,
        )
        inputs.int(
            "epochs",
            label="Epochs",
            description="Number of training epochs",
            default=_get_config_value(cfg, "training", "epochs", default=100),
        )
        inputs.int(
            "batch_size",
            label="Batch Size",
            description="Training batch size",
            default=_get_config_value(cfg, "training", "batch_size", default=16),
        )
        inputs.int(
            "imgsz",
            label="Image Size",
            description="Input image size in pixels",
            default=_get_config_value(cfg, "training", "imgsz", default=640),
        )
        inputs.float(
            "lr0",
            label="Learning Rate",
            description="Initial learning rate",
            default=_get_config_value(cfg, "training", "lr0", default=0.01),
        )
        inputs.int(
            "patience",
            label="Early Stopping Patience",
            description="Epochs to wait for improvement before stopping",
            default=_get_config_value(cfg, "training", "patience", default=50),
        )
        inputs.float(
            "split_train",
            label="Train Split",
            description="Training data split ratio",
            default=_get_config_value(cfg, "training", "split_train", default=0.8),
        )
        inputs.float(
            "split_val",
            label="Validation Split",
            description="Validation data split ratio",
            default=_get_config_value(cfg, "training", "split_val", default=0.1),
        )
        inputs.float(
            "split_test",
            label="Test Split",
            description="Test data split ratio",
            default=_get_config_value(cfg, "training", "split_test", default=0.1),
        )
        inputs.str(
            "extra_hyperparameters",
            label="Extra Hyperparameters",
            description="Additional hyperparameters as JSON (e.g., {\"warmup_epochs\": 5})",
            required=False,
        )

        # === Environment Variables ===
        inputs.str(
            "wandb_project",
            label="W&B Project",
            description="W&B project name (optional)",
            default=_get_config_value(cfg, "wandb", "project"),
            required=False,
        )
        inputs.str(
            "extra_environment",
            label="Extra Environment Variables",
            description="Additional env vars as JSON (e.g., {\"MY_VAR\": \"value\"})",
            required=False,
        )

        # === AWS Configuration ===
        # NOTE: AWS credentials are resolved from FiftyOne Secrets Manager (Teams)
        # or environment variables (open-source). They are not passed through the UI.
        inputs.str(
            "region",
            label="AWS Region",
            description="AWS region",
            default=_get_config_value(cfg, "aws", "region", default="us-east-1"),
        )
        inputs.str(
            "profile",
            label="AWS Profile",
            description="AWS profile name (optional, for local development)",
            default=_get_config_value(cfg, "aws", "profile"),
            required=False,
        )

        return types.Property(inputs)

    def execute(self, ctx):
        """Execute SageMaker training job.

        Supports two split modes:
        - ratio: Export current view, split inside SageMaker container
        - saved_views: Export each saved view as a separate split, upload pre-split

        Returns:
            dict: Job launch results including job_name and status.
        """
        params = ctx.params
        dataset = ctx.dataset

        logger.info(f"Launching SageMaker training for dataset: {dataset.name}")

        # Resolve credentials from secrets (never from params)
        creds = _resolve_aws_credentials(ctx)
        wandb_api_key = _resolve_wandb_api_key(ctx)

        # Build SageMaker configuration
        sm_config = SageMakerConfig(
            role=params["role"],
            bucket=params["bucket"],
            image_uri=params["image_uri"],
            instance_type=params["instance_type"],
            instance_count=params["instance_count"],
            subnets=_parse_list(params.get("subnets")),
            security_group_ids=_parse_list(params.get("security_group_ids")),
            region=params.get("region", "us-east-1"),
            profile=params.get("profile") or None,
            aws_access_key_id=creds.get("aws_access_key_id"),
            aws_secret_access_key=creds.get("aws_secret_access_key"),
            tags=_parse_tags(params.get("tags")),
            output_path=params.get("output_path") or None,
            code_location=params.get("code_location") or None,
            base_job_name=params.get("base_job_name", "yolo26-training"),
        )

        split_mode = params.get("split_mode", "ratio")
        label_field = params.get("label_field", "ground_truth")
        pre_split = split_mode in ("saved_views", "tags")

        # Build training job configuration
        job_config = TrainingJobConfig(
            dataset_s3_uri="",  # Will be set after upload
            model=params.get("model", "yolo26n.pt"),
            epochs=params.get("epochs", 100),
            batch_size=params.get("batch_size", 16),
            imgsz=params.get("imgsz", 640),
            patience=params.get("patience", 50),
            lr0=params.get("lr0", 0.01),
            split_train=params.get("split_train", 0.8),
            split_val=params.get("split_val", 0.1),
            split_test=params.get("split_test", 0.1),
            pre_split=pre_split,
            wandb_project=params.get("wandb_project") or None,
            wandb_api_key=wandb_api_key,
            extra_hyperparameters=_parse_json_dict(params.get("extra_hyperparameters")),
            extra_environment=_parse_json_dict(params.get("extra_environment")),
        )

        ctx.set_progress(progress=0.1, label="Exporting dataset to YOLO format...")

        # Export dataset to temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            export_dir = Path(tmp_dir) / "dataset"

            # Auto-detect classes from full dataset for consistency
            det_label_field = f"{label_field}.detections.label"
            classes = dataset.distinct(det_label_field)

            if split_mode == "tags":
                # Tags mode: filter by sample tags
                train_tag = params.get("train_tag", "train")
                val_tag = params.get("val_tag", "val")
                test_tag = params.get("test_tag")

                view = ctx.view if hasattr(ctx, "view") and ctx.view is not None else dataset

                train_samples = view.match_tags(train_tag)
                if len(train_samples) == 0:
                    return {
                        "status": "error",
                        "message": f"No samples found with tag '{train_tag}'",
                    }
                _export_dataset_to_yolo(
                    dataset=train_samples,
                    export_dir=export_dir,
                    label_field=label_field,
                    classes=classes,
                    split="train",
                )

                val_samples = view.match_tags(val_tag)
                if len(val_samples) == 0:
                    return {
                        "status": "error",
                        "message": f"No samples found with tag '{val_tag}'",
                    }
                _export_dataset_to_yolo(
                    dataset=val_samples,
                    export_dir=export_dir,
                    label_field=label_field,
                    classes=classes,
                    split="val",
                )

                if test_tag:
                    test_samples = view.match_tags(test_tag)
                    if len(test_samples) > 0:
                        _export_dataset_to_yolo(
                            dataset=test_samples,
                            export_dir=export_dir,
                            label_field=label_field,
                            classes=classes,
                            split="test",
                        )

            elif split_mode == "saved_views":
                # Export each saved view as a separate split
                train_view_name = params.get("train_view")
                val_view_name = params.get("val_view")
                test_view_name = params.get("test_view")

                if not train_view_name or not val_view_name:
                    return {
                        "status": "error",
                        "message": "Train and validation views are required in saved_views mode",
                    }

                # Export train split
                train_view = dataset.load_saved_view(train_view_name)
                _export_dataset_to_yolo(
                    dataset=train_view,
                    export_dir=export_dir,
                    label_field=label_field,
                    classes=classes,
                    split="train",
                )

                # Export val split
                val_view = dataset.load_saved_view(val_view_name)
                _export_dataset_to_yolo(
                    dataset=val_view,
                    export_dir=export_dir,
                    label_field=label_field,
                    classes=classes,
                    split="val",
                )

                # Export test split if provided
                if test_view_name:
                    test_view = dataset.load_saved_view(test_view_name)
                    _export_dataset_to_yolo(
                        dataset=test_view,
                        export_dir=export_dir,
                        label_field=label_field,
                        classes=classes,
                        split="test",
                    )
            else:
                # Ratio mode: export the active view (ctx.view), split in SageMaker
                view = ctx.view if hasattr(ctx, "view") and ctx.view is not None else dataset
                _export_dataset_to_yolo(
                    dataset=view,
                    export_dir=export_dir,
                    label_field=label_field,
                    classes=classes,
                    split="val",
                )

            ctx.set_progress(progress=0.3, label="Uploading dataset to S3...")

            # Create SageMaker client and upload dataset
            client = SageMakerClient(sm_config)

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            dataset_name = f"{dataset.name}-{timestamp}"

            s3_uri = client.upload_dataset(
                local_path=export_dir,
                s3_prefix="datasets",
                dataset_name=dataset_name,
            )

            # Update job config with S3 URI
            job_config.dataset_s3_uri = s3_uri

            ctx.set_progress(progress=0.5, label="Launching SageMaker training job...")

            # Launch training job
            job_name = client.create_training_job(
                job_config=job_config,
                wait=False,  # Don't block - let user monitor separately
            )

            logger.info(f"Training job launched: {job_name}")

        return {
            "status": "success",
            "job_name": job_name,
            "dataset_s3_uri": s3_uri,
            "message": f"Training job '{job_name}' launched successfully. Use 'Get Training Job Status' to monitor progress.",
        }


class GetTrainingJobStatus(foo.Operator):
    """Operator to get the status of a SageMaker training job."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="get_training_job_status",
            label="Get Training Job Status",
            description="Check the status of a SageMaker training job",
            unlisted=False,
            allow_immediate_execution=True,
        )

    def resolve_input(self, ctx):
        """Define the input form for the operator."""
        inputs = types.Object()

        inputs.str(
            "job_name",
            label="Job Name",
            description="SageMaker training job name",
            required=True,
        )
        inputs.str(
            "bucket",
            label="S3 Bucket",
            description="S3 bucket used for training",
            required=True,
        )
        inputs.str(
            "region",
            label="AWS Region",
            description="AWS region",
            default="us-east-1",
        )
        inputs.str(
            "profile",
            label="AWS Profile",
            description="AWS profile name (optional)",
            required=False,
        )

        return types.Property(inputs)

    def execute(self, ctx):
        """Get training job status.

        Returns:
            dict: Job status information.
        """
        params = ctx.params

        sm_config = _build_sagemaker_config_from_params(params, ctx=ctx)
        client = SageMakerClient(sm_config)
        status = client.get_job_status(params["job_name"])

        logger.info(f"Job status for {params['job_name']}: {status['status']}")

        return {
            "status": "success",
            "job_status": status,
        }


class ListTrainingJobs(foo.Operator):
    """Operator to list recent SageMaker training jobs."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="list_training_jobs",
            label="List Training Jobs",
            description="List recent SageMaker training jobs",
            unlisted=False,
            allow_immediate_execution=True,
        )

    def resolve_input(self, ctx):
        """Define the input form for the operator."""
        inputs = types.Object()

        inputs.str(
            "bucket",
            label="S3 Bucket",
            description="S3 bucket used for training",
            required=True,
        )
        inputs.str(
            "name_contains",
            label="Name Filter",
            description="Filter jobs by name substring",
            required=False,
        )
        inputs.str(
            "status_filter",
            label="Status Filter",
            description="Filter by status (InProgress, Completed, Failed, Stopped)",
            required=False,
        )
        inputs.int(
            "max_results",
            label="Max Results",
            description="Maximum number of jobs to return",
            default=10,
        )
        inputs.str(
            "region",
            label="AWS Region",
            description="AWS region",
            default="us-east-1",
        )
        inputs.str(
            "profile",
            label="AWS Profile",
            description="AWS profile name (optional)",
            required=False,
        )

        return types.Property(inputs)

    def execute(self, ctx):
        """List training jobs.

        Returns:
            dict: List of training job summaries.
        """
        params = ctx.params

        sm_config = _build_sagemaker_config_from_params(params, ctx=ctx)
        client = SageMakerClient(sm_config)
        jobs = client.list_jobs(
            name_contains=params.get("name_contains"),
            status_equals=params.get("status_filter"),
            max_results=params.get("max_results", 10),
        )

        logger.info(f"Found {len(jobs)} training jobs")

        return {
            "status": "success",
            "jobs": jobs,
            "count": len(jobs),
        }


class DownloadModelArtifacts(foo.Operator):
    """Operator to download model artifacts from a completed training job."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="download_model_artifacts",
            label="Download Model Artifacts",
            description="Download trained model from a completed SageMaker job",
            unlisted=False,
            allow_immediate_execution=False,
            allow_delegated_execution=True,
            default_choice_to_delegated=True,
        )

    def resolve_input(self, ctx):
        """Define the input form for the operator."""
        inputs = types.Object()

        inputs.str(
            "job_name",
            label="Job Name",
            description="SageMaker training job name",
            required=True,
        )
        inputs.str(
            "bucket",
            label="S3 Bucket",
            description="S3 bucket used for training",
            required=True,
        )
        inputs.str(
            "download_path",
            label="Download Path",
            description="Local path to download artifacts to",
            required=True,
        )
        inputs.bool(
            "extract",
            label="Extract Archive",
            description="Extract the model.tar.gz archive",
            default=True,
        )
        inputs.str(
            "region",
            label="AWS Region",
            description="AWS region",
            default="us-east-1",
        )
        inputs.str(
            "profile",
            label="AWS Profile",
            description="AWS profile name (optional)",
            required=False,
        )

        return types.Property(inputs)

    def execute(self, ctx):
        """Download model artifacts.

        Returns:
            dict: Download result with path to artifacts.
        """
        params = ctx.params

        ctx.set_progress(progress=0.1, label="Connecting to SageMaker...")

        sm_config = _build_sagemaker_config_from_params(params, ctx=ctx)
        client = SageMakerClient(sm_config)

        ctx.set_progress(progress=0.3, label="Downloading model artifacts...")

        download_path = Path(params["download_path"])
        artifacts_path = client.download_artifacts(
            job_name=params["job_name"],
            local_path=download_path,
            extract=params.get("extract", True),
        )

        logger.info(f"Downloaded artifacts to {artifacts_path}")

        return {
            "status": "success",
            "artifacts_path": str(artifacts_path),
            "message": f"Model artifacts downloaded to {artifacts_path}",
        }


class StopTrainingJob(foo.Operator):
    """Operator to stop a running SageMaker training job."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="stop_training_job",
            label="Stop Training Job",
            description="Stop a running SageMaker training job",
            unlisted=False,
            allow_immediate_execution=True,
        )

    def resolve_input(self, ctx):
        """Define the input form for the operator."""
        inputs = types.Object()

        inputs.str(
            "job_name",
            label="Job Name",
            description="SageMaker training job name to stop",
            required=True,
        )
        inputs.str(
            "bucket",
            label="S3 Bucket",
            description="S3 bucket used for training",
            required=True,
        )
        inputs.str(
            "region",
            label="AWS Region",
            description="AWS region",
            default="us-east-1",
        )
        inputs.str(
            "profile",
            label="AWS Profile",
            description="AWS profile name (optional)",
            required=False,
        )

        return types.Property(inputs)

    def execute(self, ctx):
        """Stop a training job.

        Returns:
            dict: Stop result.
        """
        params = ctx.params

        sm_config = _build_sagemaker_config_from_params(params, ctx=ctx)
        client = SageMakerClient(sm_config)
        client.stop_job(params["job_name"])

        logger.info(f"Stop request sent for job: {params['job_name']}")

        return {
            "status": "success",
            "job_name": params["job_name"],
            "message": f"Stop request sent for job '{params['job_name']}'",
        }


class ApplyYoloModel(foo.Operator):
    """Operator to apply a trained YOLO model to a FiftyOne dataset."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="apply_yolo_model",
            label="Apply YOLO Model",
            description="Run inference with a trained YOLO model on the current view",
            unlisted=False,
            allow_immediate_execution=False,
            allow_delegated_execution=True,
            default_choice_to_delegated=True,
        )

    def resolve_input(self, ctx):
        """Define the input form for the operator."""
        inputs = types.Object()

        inputs.str(
            "weight_source",
            label="Weight Source",
            description="'job' to download from completed SageMaker job, 'path' for direct path",
            default="path",
            required=True,
        )
        inputs.str(
            "job_name",
            label="Job Name",
            description="Completed SageMaker training job name (if source is 'job')",
            required=False,
        )
        inputs.str(
            "weights_path",
            label="Weights Path",
            description="Direct path to YOLO model weights (if source is 'path')",
            required=False,
        )
        inputs.str(
            "det_field",
            label="Prediction Field",
            description="Field name where predictions will be stored",
            default="predictions",
            required=True,
        )
        inputs.float(
            "confidence",
            label="Confidence Threshold",
            description="Minimum confidence for predictions",
            default=0.25,
        )

        # AWS config for downloading from job
        inputs.str(
            "bucket",
            label="S3 Bucket",
            description="S3 bucket (required if downloading from job)",
            required=False,
        )
        inputs.str(
            "region",
            label="AWS Region",
            description="AWS region",
            default="us-east-1",
        )
        inputs.str(
            "profile",
            label="AWS Profile",
            description="AWS profile name (optional)",
            required=False,
        )

        return types.Property(inputs)

    def execute(self, ctx):
        """Apply YOLO model to dataset.

        Downloads model artifacts if needed, loads YOLO model, and runs
        inference on the current dataset view.

        Returns:
            dict: Result with prediction count.
        """
        from ultralytics import YOLO

        params = ctx.params
        dataset = ctx.dataset
        view = ctx.view if hasattr(ctx, "view") and ctx.view is not None else dataset

        weight_source = params.get("weight_source", "path")
        det_field = params.get("det_field", "predictions")
        confidence = params.get("confidence", 0.25)

        ctx.set_progress(progress=0.1, label="Loading model...")

        if weight_source == "job":
            # Download from completed SageMaker job
            job_name = params.get("job_name")
            if not job_name:
                return {"status": "error", "message": "Job name is required when source is 'job'"}

            sm_config = _build_sagemaker_config_from_params(params, ctx=ctx)
            client = SageMakerClient(sm_config)

            ctx.set_progress(progress=0.2, label="Downloading model artifacts...")

            with tempfile.TemporaryDirectory() as tmp_dir:
                artifacts_path = client.download_artifacts(
                    job_name=job_name,
                    local_path=Path(tmp_dir),
                    extract=True,
                )

                # Find the best.pt or model weights file
                weights_file = _find_weights_file(artifacts_path)
                if not weights_file:
                    return {"status": "error", "message": f"No model weights found in artifacts at {artifacts_path}"}

                ctx.set_progress(progress=0.4, label="Running inference...")

                model = YOLO(str(weights_file))
                model.conf = confidence
                view.apply_model(model, label_field=det_field, num_workers=0)
        else:
            # Direct path to weights
            weights_path = params.get("weights_path")
            if not weights_path:
                return {"status": "error", "message": "Weights path is required when source is 'path'"}

            model = YOLO(weights_path)
            model.conf = confidence

            ctx.set_progress(progress=0.4, label="Running inference...")

            view.apply_model(model, label_field=det_field, num_workers=0)

        sample_count = len(view)
        logger.info(f"Applied model to {sample_count} samples, predictions in '{det_field}'")

        return {
            "status": "success",
            "sample_count": sample_count,
            "det_field": det_field,
            "message": f"Applied model to {sample_count} samples. Predictions stored in '{det_field}'.",
        }


def _find_weights_file(artifacts_path: Path) -> Optional[Path]:
    """Find YOLO model weights in extracted artifacts.

    Looks for best.pt, last.pt, or any .pt file in the artifacts directory.

    Args:
        artifacts_path: Path to extracted model artifacts.

    Returns:
        Path to weights file, or None if not found.
    """
    artifacts_path = Path(artifacts_path)

    # Priority order for weight files
    candidates = [
        "best.pt",
        "weights/best.pt",
        "last.pt",
        "weights/last.pt",
        "model.pt",
    ]

    for candidate in candidates:
        path = artifacts_path / candidate
        if path.exists():
            return path

    # Fallback: find any .pt file
    pt_files = list(artifacts_path.rglob("*.pt"))
    if pt_files:
        return pt_files[0]

    return None


class SageMakerPanel(Panel):
    """Panel for managing SageMaker YOLO training jobs."""

    @property
    def config(self):
        return PanelConfig(
            name="sagemaker_panel",
            label="SageMaker Trainer",
            icon="model_training",
            surfaces="grid",
            allow_multiple=False,
        )

    def on_load(self, ctx):
        """Initialize panel state from default config."""
        cfg = _load_default_config()

        # Load saved views from dataset
        saved_views = []
        if ctx.dataset:
            try:
                view_names = ctx.dataset.list_saved_views()
                for name in view_names:
                    try:
                        view = ctx.dataset.load_saved_view(name)
                        saved_views.append({
                            "name": name,
                            "sample_count": len(view),
                        })
                    except Exception:
                        saved_views.append({"name": name, "sample_count": 0})
            except Exception:
                pass

        # Build default S3 weights path from output config
        output_path = _get_config_value(cfg, "sagemaker", "output_path", default="")
        default_weights_path = f"{output_path}/model.tar.gz" if output_path else ""

        ctx.panel.set_state({
            "active_tab": "train",
            # AWS
            "image_uri": _get_config_value(cfg, "sagemaker", "image_uri", default=""),
            "role": _get_config_value(cfg, "sagemaker", "role", default=""),
            "bucket": _get_config_value(cfg, "sagemaker", "bucket", default=""),
            "region": _get_config_value(cfg, "aws", "region", default="us-east-1"),
            "profile": _get_config_value(cfg, "aws", "profile", default=""),
            # Instance
            "instance_type": _get_config_value(cfg, "sagemaker", "instance_type", default="ml.g4dn.xlarge"),
            "instance_count": _get_config_value(cfg, "sagemaker", "instance_count", default=1),
            "subnets": _get_config_value(cfg, "sagemaker", "subnets", default=""),
            "security_group_ids": _get_config_value(cfg, "sagemaker", "security_group_ids", default=""),
            # Training
            "task": _get_config_value(cfg, "training", "task", default="detect"),
            "model": _get_config_value(cfg, "training", "model", default="yolo26n.pt"),
            "label_field": _get_config_value(cfg, "dataset", "label_field", default="ground_truth"),
            "epochs": _get_config_value(cfg, "training", "epochs", default=100),
            "batch_size": _get_config_value(cfg, "training", "batch_size", default=16),
            "imgsz": _get_config_value(cfg, "training", "imgsz", default=640),
            "lr0": _get_config_value(cfg, "training", "lr0", default=0.01),
            "patience": _get_config_value(cfg, "training", "patience", default=50),
            # Dataset / splits
            "split_mode": "tags",
            "split_train": _get_config_value(cfg, "training", "split_train", default=0.8),
            "split_val": _get_config_value(cfg, "training", "split_val", default=0.1),
            "split_test": _get_config_value(cfg, "training", "split_test", default=0.1),
            "saved_views": saved_views,
            "train_view": "",
            "val_view": "",
            "test_view": "",
            "train_tag": "train",
            "val_tag": "val",
            "test_tag": "test",
            # W&B
            "wandb_project": _get_config_value(cfg, "wandb", "project", default=""),
            # Advanced
            "output_path": _get_config_value(cfg, "sagemaker", "output_path", default=""),
            "code_location": _get_config_value(cfg, "sagemaker", "code_location", default=""),
            "base_job_name": _get_config_value(cfg, "sagemaker", "base_job_name", default="yolo26-training"),
            "tags": _get_config_value(cfg, "sagemaker", "tags", default=""),
            "disable_profiler": _get_config_value(cfg, "sagemaker", "disable_profiler", default=True),
            "enable_sagemaker_metrics": _get_config_value(cfg, "sagemaker", "enable_sagemaker_metrics", default=False),
            # Monitor
            "jobs_list": [],
            "selected_job": "",
            "job_status": None,
            # Apply
            "weight_source": "path",
            "apply_job_name": "",
            "apply_weights_s3": "",
            "weights_path": default_weights_path,
            "det_field": "predictions",
            "confidence": 0.25,
        })

    # --- Tab Navigation ---

    def on_change_tab(self, ctx):
        tab = ctx.panel.state.active_tab
        if tab == "monitor":
            self._refresh_jobs(ctx)

    # --- Radio Change Callbacks ---

    def on_change_split_mode(self, ctx):
        """Re-render when split mode changes between tags/saved_views."""

    def on_change_weight_source(self, ctx):
        """Re-render when weight source changes between path/job."""

    # --- Train Tab Actions ---

    def on_launch_training(self, ctx):
        state = ctx.panel.state
        split_mode = state.split_mode or "ratio"
        params = {
            "image_uri": state.image_uri or "",
            "role": state.role or "",
            "bucket": state.bucket or "",
            "region": state.region or "us-east-1",
            "profile": state.profile or "",
            "instance_type": state.instance_type or "ml.g4dn.xlarge",
            "instance_count": state.instance_count or 1,
            "subnets": state.subnets or "",
            "security_group_ids": state.security_group_ids or "",
            "model": state.model or "yolo26n.pt",
            "label_field": state.label_field or "ground_truth",
            "epochs": state.epochs or 100,
            "batch_size": state.batch_size or 16,
            "imgsz": state.imgsz or 640,
            "lr0": state.get("lr0", 0.01),
            "patience": state.get("patience", 50),
            "split_train": state.get("split_train", 0.8),
            "split_val": state.get("split_val", 0.2),
            "split_test": state.get("split_test", 0.0),
            "wandb_project": state.wandb_project or "",
            "output_path": state.output_path or "",
            "code_location": state.code_location or "",
            "base_job_name": state.base_job_name or "yolo26-training",
            "tags": state.tags or "",
            "disable_profiler": state.disable_profiler if state.disable_profiler is not None else True,
            "enable_sagemaker_metrics": state.enable_sagemaker_metrics or False,
            "split_mode": split_mode,
        }
        if split_mode == "tags":
            params["train_tag"] = state.train_tag or "train"
            params["val_tag"] = state.val_tag or "val"
            params["test_tag"] = state.test_tag or ""
        elif split_mode == "saved_views":
            params["train_view"] = state.train_view or ""
            params["val_view"] = state.val_view or ""
            params["test_view"] = state.test_view or ""
        ctx.trigger(
            "@roboav8r/fiftyone-aws/launch_sagemaker_training",
            params=params,
        )

    def on_apply_split_tags(self, ctx):
        """Apply random split tags to samples in the current view."""
        state = ctx.panel.state
        ctx.trigger(
            "@roboav8r/fiftyone-aws/apply_split_tags",
            params={
                "split_train": state.get("split_train", 0.8),
                "split_val": state.get("split_val", 0.1),
                "split_test": state.get("split_test", 0.1),
            },
        )

    # --- Monitor Tab Actions ---

    def on_refresh_jobs(self, ctx):
        self._refresh_jobs(ctx)
        ctx.ops.notify("Jobs refreshed")

    def on_get_job_status(self, ctx):
        state = ctx.panel.state
        job_name = state.selected_job
        if not job_name:
            ctx.ops.notify("Please select a job first", variant="warning")
            return
        try:
            client = SageMakerClient(self._build_sm_config(ctx))
            status = client.get_job_status(job_name)
            ctx.panel.set_state("job_status", status)
        except Exception as e:
            ctx.ops.notify(f"Error getting status: {e}", variant="error")

    def on_stop_job(self, ctx):
        state = ctx.panel.state
        job_name = state.selected_job
        if not job_name:
            ctx.ops.notify("Please select a job first", variant="warning")
            return
        try:
            client = SageMakerClient(self._build_sm_config(ctx))
            client.stop_job(job_name)
            ctx.ops.notify(f"Stop request sent for '{job_name}'")
            status = client.get_job_status(job_name)
            ctx.panel.set_state("job_status", status)
        except Exception as e:
            ctx.ops.notify(f"Error stopping job: {e}", variant="error")

    # --- Apply Tab Actions ---

    def on_select_apply_job(self, ctx):
        """When a job is selected in apply tab, fetch its model artifacts path."""
        state = ctx.panel.state
        job_name = state.apply_job_name
        if not job_name:
            return
        try:
            client = SageMakerClient(self._build_sm_config(ctx))
            status = client.get_job_status(job_name)
            artifacts_uri = status.get("model_artifacts", "")
            ctx.panel.set_state("apply_weights_s3", artifacts_uri)
        except Exception as e:
            ctx.panel.set_state("apply_weights_s3", f"Error: {e}")

    def on_apply_model(self, ctx):
        state = ctx.panel.state
        params = {
            "weight_source": state.weight_source or "path",
            "job_name": state.apply_job_name or "",
            "weights_path": state.weights_path or "",
            "det_field": state.det_field or "predictions",
            "confidence": state.get("confidence", 0.25),
            "bucket": state.bucket or "",
            "region": state.region or "us-east-1",
            "profile": state.profile or "",
        }
        ctx.trigger(
            "@roboav8r/fiftyone-aws/apply_yolo_model",
            params=params,
        )

    # --- Helpers ---

    def _build_sm_config(self, ctx):
        """Build SageMakerConfig from panel state and ctx.secrets."""
        state = ctx.panel.state
        creds = _resolve_aws_credentials(ctx)
        return SageMakerConfig(
            role="",
            bucket=state.bucket or "",
            image_uri="",
            region=state.region or "us-east-1",
            profile=state.profile if state.profile else None,
            aws_access_key_id=creds.get("aws_access_key_id"),
            aws_secret_access_key=creds.get("aws_secret_access_key"),
        )

    def _refresh_jobs(self, ctx):
        state = ctx.panel.state
        try:
            client = SageMakerClient(self._build_sm_config(ctx))
            jobs = client.list_jobs(max_results=20)
            ctx.panel.set_state("jobs_list", jobs)
        except Exception as e:
            ctx.ops.notify(f"Error loading jobs: {e}", variant="error")
            ctx.panel.set_state("jobs_list", [])

    # --- Rendering ---

    def render(self, ctx):
        panel = types.Object()
        active_tab = ctx.panel.state.active_tab or "train"

        # Tab navigation
        tabs_view = types.TabsView()
        tabs_view.add_choice("train", label="Train")
        tabs_view.add_choice("monitor", label="Monitor")
        tabs_view.add_choice("apply", label="Apply Model")
        panel.enum(
            "active_tab", tabs_view.values(),
            default=active_tab, view=tabs_view,
            on_change=self.on_change_tab,
        )

        if active_tab == "train":
            self._render_train_tab(ctx, panel)
        elif active_tab == "monitor":
            self._render_monitor_tab(ctx, panel)
        elif active_tab == "apply":
            self._render_apply_tab(ctx, panel)

        return types.Property(panel, view=types.GridView(gap=2, pad=2))

    def _render_train_tab(self, ctx, panel):
        state = ctx.panel.state

        # View info header - dynamically shows current view name and count
        view = ctx.view if hasattr(ctx, "view") and ctx.view is not None else ctx.dataset
        view_sample_count = len(view) if view is not None else 0
        if ctx.dataset is not None and view_sample_count != len(ctx.dataset):
            view_name = "Current View"
        else:
            view_name = ctx.dataset.name if ctx.dataset else "No Dataset"
        panel.view(
            "view_info",
            types.TextView(
                title=f"{view_name}: {view_sample_count} samples",
                variant="body1",
                bold=True,
            ),
        )

        # --- Dataset & Splits ---
        panel.view("split_header", types.TextView(title="Dataset & Splits", variant="h6", bold=True))

        # Split ratio fields (always visible)
        panel.float("split_train", label="Train Split", default=state.get("split_train", 0.8))
        panel.float("split_val", label="Validation Split", default=state.get("split_val", 0.1))
        panel.float("split_test", label="Test Split", default=state.get("split_test", 0.1), min=0.0)

        # Apply Split Tags button
        panel.btn(
            "apply_split_tags_btn",
            label="Apply Split Tags",
            on_click=self.on_apply_split_tags,
            variant="outlined",
        )
        panel.view("apply_tags_hint", types.TextView(
            title="Clears existing train/val/test tags and applies new random splits to the current view",
            variant="body2", color="text.secondary",
        ))

        # Split mode radio: Tags / Saved Views
        split_mode_radio = types.RadioGroup()
        split_mode_radio.add_choice("tags", label="Tags")
        split_mode_radio.add_choice("saved_views", label="Saved Views")
        panel.enum(
            "split_mode", split_mode_radio.values(),
            default=state.split_mode or "tags",
            label="Split Mode", view=split_mode_radio,
            on_change=self.on_change_split_mode,
        )

        split_mode = state.split_mode or "tags"

        if split_mode == "tags":
            # Get tag counts from the current view
            tag_counts = {}
            if view is not None:
                try:
                    tag_counts = view.count_sample_tags()
                except Exception:
                    pass

            if tag_counts:
                tag_names = list(tag_counts.keys())

                # Train tag dropdown
                train_tag_dropdown = types.AutocompleteView()
                for tag_name in tag_names:
                    count = tag_counts.get(tag_name, 0)
                    train_tag_dropdown.add_choice(tag_name, label=f"{tag_name} ({count} samples)")
                panel.enum(
                    "train_tag", train_tag_dropdown.values(),
                    default=state.get("train_tag", "train"),
                    label="Train Data *", view=train_tag_dropdown,
                )

                # Val tag dropdown
                val_tag_dropdown = types.AutocompleteView()
                for tag_name in tag_names:
                    count = tag_counts.get(tag_name, 0)
                    val_tag_dropdown.add_choice(tag_name, label=f"{tag_name} ({count} samples)")
                panel.enum(
                    "val_tag", val_tag_dropdown.values(),
                    default=state.get("val_tag", "val"),
                    label="Val Data *", view=val_tag_dropdown,
                )

                # Test tag dropdown (optional)
                test_tag_dropdown = types.AutocompleteView()
                test_tag_dropdown.add_choice("", label="(none)")
                for tag_name in tag_names:
                    count = tag_counts.get(tag_name, 0)
                    test_tag_dropdown.add_choice(tag_name, label=f"{tag_name} ({count} samples)")
                panel.enum(
                    "test_tag", test_tag_dropdown.values(),
                    default=state.get("test_tag", "test"),
                    label="Test Data (optional)", view=test_tag_dropdown,
                )
            else:
                panel.view("no_tags", types.TextView(
                    title="No sample tags found. Use 'Apply Split Tags' above to create train/val/test tags.",
                    variant="body2", color="text.secondary",
                ))

        elif split_mode == "saved_views":
            saved_views = state.saved_views or []
            if saved_views:
                for field_key, field_label, required in [
                    ("train_view", "Train View *", True),
                    ("val_view", "Val View *", True),
                    ("test_view", "Test View (optional)", False),
                ]:
                    sv_dropdown = types.AutocompleteView()
                    if not required:
                        sv_dropdown.add_choice("", label="(none)")
                    for sv in saved_views:
                        sv_name = sv.get("name", "")
                        sv_count = sv.get("sample_count", 0)
                        sv_dropdown.add_choice(sv_name, label=f"{sv_name} ({sv_count} samples)")
                    panel.enum(
                        field_key, sv_dropdown.values(),
                        default=state.get(field_key, ""),
                        label=field_label, view=sv_dropdown,
                    )
            else:
                panel.view("no_views", types.TextView(
                    title="No saved views found. Create saved views first or use Tags mode.",
                    variant="body2", color="text.secondary",
                ))

        # --- AWS Configuration ---
        panel.view("aws_header", types.TextView(title="AWS Configuration", variant="h6", bold=True))
        panel.view("aws_creds_note", types.TextView(
            title="AWS credentials are configured via Secrets Manager (Teams) or environment variables (open-source)",
            variant="body2", color="text.secondary",
        ))
        panel.str("image_uri", label="ECR Image URI", default=state.image_uri or "", required=True)
        panel.str("role", label="IAM Role ARN", default=state.role or "", required=True)
        panel.str("bucket", label="S3 Bucket", default=state.bucket or "", required=True)
        panel.str("region", label="AWS Region", default=state.region or "us-east-1")
        panel.str("profile", label="AWS Profile", default=state.profile or "")

        # --- Instance Configuration ---
        panel.view("instance_header", types.TextView(title="Instance Configuration", variant="h6", bold=True))
        instance_type_view = types.AutocompleteView()
        for itype in [
            "ml.g4dn.xlarge", "ml.g4dn.2xlarge", "ml.g4dn.4xlarge",
            "ml.g5.xlarge", "ml.g5.2xlarge",
            "ml.p3.2xlarge", "ml.p3.8xlarge", "ml.p4d.24xlarge",
        ]:
            instance_type_view.add_choice(itype, label=itype)
        panel.enum(
            "instance_type", instance_type_view.values(),
            default=state.instance_type or "ml.g4dn.xlarge",
            label="Instance Type", view=instance_type_view,
        )
        panel.int("instance_count", label="Instance Count", default=state.instance_count or 1, min=1)
        panel.str("subnets", label="VPC Subnets (comma-separated, optional)", default=state.subnets or "")
        panel.str("security_group_ids", label="Security Groups (comma-separated, optional)", default=state.security_group_ids or "")

        # --- Training Configuration ---
        panel.view("train_header", types.TextView(title="Training Configuration", variant="h6", bold=True))

        # Task type: detection or segmentation
        task_radio = types.RadioGroup()
        task_radio.add_choice("detect", label="Detection")
        task_radio.add_choice("segment", label="Segmentation")
        panel.enum(
            "task", task_radio.values(),
            default=state.task or "detect",
            label="Task", view=task_radio,
        )

        # Model selector based on task
        task = state.task or "detect"
        model_view = types.AutocompleteView()
        if task == "segment":
            seg_models = [
                "yolo26n-seg.pt", "yolo26s-seg.pt", "yolo26m-seg.pt",
                "yolo26l-seg.pt", "yolo26x-seg.pt",
            ]
            for m in seg_models:
                model_view.add_choice(m, label=m)
            current_model = state.model or "yolo26n-seg.pt"
            if current_model not in seg_models:
                current_model = "yolo26n-seg.pt"
        else:
            det_models = [
                "yolo26n.pt", "yolo26s.pt", "yolo26m.pt",
                "yolo26l.pt", "yolo26x.pt",
            ]
            for m in det_models:
                model_view.add_choice(m, label=m)
            current_model = state.model or "yolo26n.pt"
            if current_model not in det_models:
                current_model = "yolo26n.pt"
        panel.enum(
            "model", model_view.values(),
            default=current_model,
            label="YOLO Model", view=model_view,
        )

        panel.str("label_field", label="Label Field", default=state.label_field or "ground_truth")
        panel.int("epochs", label="Epochs", default=state.epochs or 100, min=1)
        panel.int("batch_size", label="Batch Size", default=state.batch_size or 16, min=1)
        panel.int("imgsz", label="Image Size", default=state.get("imgsz", 640), min=32)
        panel.float("lr0", label="Learning Rate", default=state.get("lr0", 0.01))
        panel.int("patience", label="Early Stopping Patience", default=state.get("patience", 50), min=0)

        # --- W&B ---
        panel.view("wandb_header", types.TextView(title="Weights & Biases (Optional)", variant="h6", bold=True))
        panel.view("wandb_creds_note", types.TextView(
            title="W&B API key is configured via Secrets Manager (Teams) or WANDB_API_KEY environment variable",
            variant="body2", color="text.secondary",
        ))
        panel.str("wandb_project", label="W&B Project", default=state.wandb_project or "")

        # --- Advanced ---
        panel.view("adv_header", types.TextView(title="Advanced Settings", variant="h6", bold=True))
        panel.str("output_path", label="Output S3 Path", default=state.output_path or "")
        panel.str("code_location", label="Code S3 Path", default=state.code_location or "")
        panel.str("base_job_name", label="Base Job Name", default=state.base_job_name or "yolo26-training")
        panel.str("tags", label="Tags (key=value,...)", default=state.tags or "")
        panel.bool("disable_profiler", label="Disable Profiler", default=state.disable_profiler if state.disable_profiler is not None else True)
        panel.bool("enable_sagemaker_metrics", label="Enable SageMaker Metrics", default=state.enable_sagemaker_metrics or False)

        # Launch button
        panel.btn(
            "launch_btn",
            label="Launch Training Job",
            on_click=self.on_launch_training,
            variant="contained",
        )

    def _render_monitor_tab(self, ctx, panel):
        state = ctx.panel.state

        # Header with refresh
        header = panel.h_stack("monitor_header", gap=2, align_y="center")
        header.view("monitor_title", types.TextView(title="Training Jobs", variant="h6", bold=True))
        header.btn("refresh_btn", label="Refresh Jobs", on_click=self.on_refresh_jobs)

        # Job selector
        jobs_list = state.jobs_list or []
        if jobs_list:
            job_view = types.AutocompleteView()
            for job in jobs_list:
                name = job.get("job_name", "")
                jstatus = job.get("status", "")
                job_view.add_choice(name, label=f"{name} ({jstatus})")
            panel.enum(
                "selected_job", job_view.values(),
                default=state.selected_job or "",
                label="Select Job", view=job_view,
            )
            panel.btn("get_status_btn", label="Get Status", on_click=self.on_get_job_status)
        else:
            panel.view(
                "no_jobs",
                types.TextView(
                    title="No jobs found. Click 'Refresh Jobs' to load.",
                    variant="body2", color="text.secondary",
                ),
            )

        # Job status display
        job_status = state.job_status
        if job_status:
            panel.view("status_divider", types.TextView(title="Job Details", variant="h6", bold=True))

            sm_status = job_status.get("status", "Unknown")
            panel.view("status_value", types.TextView(
                title=f"Status: {sm_status}", variant="body1", bold=True,
            ))

            secondary = job_status.get("secondary_status", "")
            if secondary:
                panel.view("secondary_status", types.TextView(
                    title=f"Details: {secondary}", variant="body2",
                ))

            for key, label in [
                ("creation_time", "Created"),
                ("training_start_time", "Started"),
                ("training_end_time", "Ended"),
            ]:
                val = job_status.get(key, "")
                if val:
                    panel.view(key, types.TextView(
                        title=f"{label}: {val}", variant="body2",
                    ))

            billable = job_status.get("billable_seconds")
            if billable:
                panel.view("billable", types.TextView(
                    title=f"Billable Seconds: {billable}", variant="body2",
                ))

            model_artifacts = job_status.get("model_artifacts", "")
            if model_artifacts:
                panel.view("artifacts", types.TextView(
                    title=f"Model Artifacts: {model_artifacts}", variant="body2",
                ))

            failure = job_status.get("failure_reason")
            if failure:
                panel.view("failure", types.TextView(
                    title=f"Failure: {failure}", variant="body2",
                ))

            # Metrics
            metrics = job_status.get("metrics", {})
            if metrics:
                panel.view("metrics_header", types.TextView(
                    title="Final Metrics", variant="subtitle1", bold=True,
                ))
                for i, (metric_name, metric_val) in enumerate(metrics.items()):
                    panel.view(f"metric_{i}", types.TextView(
                        title=f"  {metric_name}: {metric_val}",
                        variant="body2",
                    ))

            # W&B info
            hyperparams = job_status.get("hyperparameters", {})
            environment = job_status.get("environment", {})
            wandb_project = hyperparams.get("wandb_project") or environment.get("WANDB_PROJECT", "")
            if wandb_project:
                panel.view("wandb_info", types.TextView(
                    title=f"W&B Project: {wandb_project}", variant="body2",
                ))

            # Stop button for InProgress jobs
            if sm_status == "InProgress":
                panel.btn(
                    "stop_btn", label="Stop Job",
                    on_click=self.on_stop_job, variant="outlined",
                )

    def _render_apply_tab(self, ctx, panel):
        state = ctx.panel.state

        panel.view("apply_title", types.TextView(
            title="Apply Trained Model", variant="h6", bold=True,
        ))
        panel.view("apply_desc", types.TextView(
            title="Run inference with a trained YOLO model on the current dataset view.",
            variant="body2", color="text.secondary",
        ))

        # Weight source
        source_radio = types.RadioGroup()
        source_radio.add_choice("path", label="Direct Path")
        source_radio.add_choice("job", label="From SageMaker Job")
        panel.enum(
            "weight_source", source_radio.values(),
            default=state.weight_source or "path",
            label="Weight Source", view=source_radio,
            on_change=self.on_change_weight_source,
        )

        weight_source = state.weight_source or "path"
        if weight_source == "job":
            # Job selector populated from same jobs_list
            jobs_list = state.jobs_list or []
            completed_jobs = [j for j in jobs_list if j.get("status") == "Completed"]
            if completed_jobs:
                job_view = types.AutocompleteView()
                for job in completed_jobs:
                    name = job.get("job_name", "")
                    job_view.add_choice(name, label=name)
                panel.enum(
                    "apply_job_name", job_view.values(),
                    default=state.apply_job_name or "",
                    label="Select Completed Job", view=job_view,
                    on_change=self.on_select_apply_job,
                )
            else:
                panel.str("apply_job_name", label="Completed Job Name", default=state.apply_job_name or "")
                panel.view("no_completed", types.TextView(
                    title="Tip: Switch to Monitor tab and click 'Refresh Jobs' to load completed jobs.",
                    variant="body2", color="text.secondary",
                ))

            # Show the resolved S3 artifacts path
            apply_weights_s3 = state.apply_weights_s3 or ""
            if apply_weights_s3:
                panel.view("apply_artifacts_path", types.TextView(
                    title=f"Model Artifacts: {apply_weights_s3}", variant="body2",
                ))
        else:
            panel.str(
                "weights_path",
                label="Path to Model Weights (local .pt or S3 URI)",
                default=state.weights_path or "",
            )

        # Inference params
        panel.str("det_field", label="Prediction Field", default=state.det_field or "predictions")
        panel.float("confidence", label="Confidence Threshold", default=state.get("confidence", 0.25))

        # Apply button
        panel.btn(
            "apply_btn", label="Apply Model",
            on_click=self.on_apply_model, variant="contained",
        )


def register(plugin):
    """Register plugin operators and panels with FiftyOne."""
    plugin.register(LaunchSageMakerTraining)
    plugin.register(GetTrainingJobStatus)
    plugin.register(ListTrainingJobs)
    plugin.register(DownloadModelArtifacts)
    plugin.register(StopTrainingJob)
    plugin.register(ApplySplitTags)
    plugin.register(ApplyYoloModel)
    plugin.register(SageMakerPanel)
