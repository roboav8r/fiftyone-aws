"""
SageMaker Integration Utilities for YOLO Training.

This module provides utilities for:
- Uploading datasets to S3
- Creating and launching SageMaker training jobs
- Monitoring job status
- Downloading model artifacts

Designed for classified AWS networks with custom ECR endpoints and VPC isolation.
"""

import json
import logging
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def _serialize_datetime(obj: Any) -> Any:
    """Convert datetime objects to ISO format strings.

    Args:
        obj: Value to serialize.

    Returns:
        ISO format string if datetime, otherwise original value.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


@dataclass
class SageMakerConfig:
    """Configuration for SageMaker training jobs.

    Attributes:
        role: IAM execution role ARN for SageMaker.
        bucket: S3 bucket for training data and artifacts.
        image_uri: Custom ECR image URI (required for classified networks).
        instance_type: EC2 instance type for training.
        instance_count: Number of training instances.
        subnets: VPC subnet IDs (required for classified networks).
        security_group_ids: VPC security group IDs.
        region: AWS region.
        profile: AWS profile name (for local development).
        aws_access_key_id: AWS access key ID (for explicit credentials).
        aws_secret_access_key: AWS secret access key (for explicit credentials).
        tags: Resource tags as key-value pairs.
        output_path: S3 path for model outputs.
        code_location: S3 path for uploaded code.
        base_job_name: Prefix for training job names.
        max_runtime_seconds: Maximum training time in seconds.
        enable_network_isolation: Whether to enable network isolation.
    """
    role: str
    bucket: str
    image_uri: str
    instance_type: str = "ml.g4dn.xlarge"
    instance_count: int = 1
    subnets: Optional[List[str]] = None
    security_group_ids: Optional[List[str]] = None
    region: str = "us-east-1"
    profile: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    output_path: Optional[str] = None
    code_location: Optional[str] = None
    base_job_name: str = "yolo26-training"
    max_runtime_seconds: int = 86400  # 24 hours
    enable_network_isolation: bool = False



@dataclass
class TrainingJobConfig:
    """Configuration for a specific training job.

    Attributes:
        dataset_s3_uri: S3 URI for training dataset.
        model: YOLO model variant (yolo26n.pt, yolo26s.pt, etc.).
        epochs: Number of training epochs.
        batch_size: Training batch size.
        imgsz: Input image size.
        patience: Early stopping patience.
        lr0: Initial learning rate.
        split_train: Training split ratio.
        split_val: Validation split ratio.
        split_test: Test split ratio.
        pre_split: Whether the data is pre-split into train/val/test dirs.
        wandb_project: W&B project name (optional).
        wandb_api_key: W&B API key (optional).
        extra_hyperparameters: Additional hyperparameters.
        extra_environment: Additional environment variables.
    """
    dataset_s3_uri: str
    model: str = "yolo26n.pt"
    epochs: int = 100
    batch_size: int = 16
    imgsz: int = 640
    patience: int = 50
    lr0: float = 0.01
    split_train: float = 0.8
    split_val: float = 0.1
    split_test: float = 0.1
    pre_split: bool = False
    wandb_project: Optional[str] = None
    wandb_api_key: Optional[str] = None
    extra_hyperparameters: Dict[str, Any] = field(default_factory=dict)
    extra_environment: Dict[str, str] = field(default_factory=dict)

    def to_hyperparameters(self) -> Dict[str, str]:
        """Convert to SageMaker hyperparameters dict.

        Returns:
            Dict with string keys and values for SageMaker.
        """
        params = {
            "model": self.model,
            "epochs": str(self.epochs),
            "batch-size": str(self.batch_size),
            "imgsz": str(self.imgsz),
            "patience": str(self.patience),
            "lr0": str(self.lr0),
            "split-train": str(self.split_train),
            "split-val": str(self.split_val),
            "split-test": str(self.split_test),
        }

        if self.pre_split:
            params["pre-split"] = "true"

        if self.wandb_project:
            params["wandb-project"] = self.wandb_project

        # Add extra hyperparameters
        for key, value in self.extra_hyperparameters.items():
            params[key] = str(value)

        return params

    def to_environment(self) -> Dict[str, str]:
        """Convert to SageMaker environment dict.

        Returns:
            Dict with environment variables for the training container.
        """
        env = {}

        if self.wandb_api_key:
            env["WANDB_API_KEY"] = self.wandb_api_key

        if self.wandb_project:
            env["WANDB_PROJECT"] = self.wandb_project

        # Add extra environment variables
        env.update(self.extra_environment)

        return env


class SageMakerClient:
    """Client for interacting with AWS SageMaker.

    Handles dataset upload, training job creation, monitoring, and artifact download.
    """

    def __init__(self, config: SageMakerConfig):
        """Initialize SageMaker client.

        Args:
            config: SageMaker configuration.
        """
        self.config = config

        # Create boto3 session
        session_kwargs = {"region_name": config.region}
        if config.profile:
            session_kwargs["profile_name"] = config.profile
        if config.aws_access_key_id and config.aws_secret_access_key:
            session_kwargs["aws_access_key_id"] = config.aws_access_key_id
            session_kwargs["aws_secret_access_key"] = config.aws_secret_access_key

        self.session = boto3.Session(**session_kwargs)
        self.s3_client = self.session.client("s3")
        self.sagemaker_client = self.session.client("sagemaker")

        logger.info(f"SageMaker client initialized for region {config.region}")

    def upload_dataset(
        self,
        local_path: Path,
        s3_prefix: str = "datasets",
        dataset_name: Optional[str] = None,
    ) -> str:
        """Upload a local dataset to S3.

        Args:
            local_path: Path to local dataset directory (YOLO format).
            s3_prefix: S3 prefix for datasets.
            dataset_name: Name for the dataset (default: directory name + timestamp).

        Returns:
            S3 URI for the uploaded dataset.

        Raises:
            FileNotFoundError: If local_path doesn't exist.
            ClientError: If S3 upload fails.
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {local_path}")

        # Generate dataset name if not provided
        if not dataset_name:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            dataset_name = f"{local_path.name}-{timestamp}"

        s3_key_prefix = f"{s3_prefix}/{dataset_name}"

        logger.info(f"Uploading dataset from {local_path} to s3://{self.config.bucket}/{s3_key_prefix}")

        # Upload all files
        file_count = 0
        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_path)
                s3_key = f"{s3_key_prefix}/{relative_path}"

                self.s3_client.upload_file(
                    str(file_path),
                    self.config.bucket,
                    s3_key,
                )
                file_count += 1

        s3_uri = f"s3://{self.config.bucket}/{s3_key_prefix}"
        logger.info(f"Uploaded {file_count} files to {s3_uri}")

        return s3_uri

    def create_training_job(
        self,
        job_config: TrainingJobConfig,
        job_name: Optional[str] = None,
        wait: bool = False,
    ) -> str:
        """Create and start a SageMaker training job using PyTorch Estimator.

        Uses the SageMaker Python SDK's PyTorch Estimator which properly handles
        source code packaging and upload, matching the client's existing workflow.

        Args:
            job_config: Training job configuration.
            job_name: Custom job name (default: auto-generated).
            wait: Whether to wait for job completion.

        Returns:
            Training job name.

        Raises:
            ClientError: If job creation fails.
        """
        import sagemaker
        from sagemaker.pytorch import PyTorch

        # Create SageMaker session
        sagemaker_session = sagemaker.Session(
            boto_session=self.session,
            default_bucket=self.config.bucket,
        )

        # Generate job name if not provided
        if not job_name:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            job_name = f"{self.config.base_job_name}-{timestamp}"

        # Get the scripts directory path
        scripts_dir = str(Path(__file__).parent / "scripts")

        # Build output path
        output_path = self.config.output_path or f"s3://{self.config.bucket}/output"
        code_location = self.config.code_location or f"s3://{self.config.bucket}/code"

        # Build estimator kwargs
        estimator_kwargs = {
            "entry_point": "train.py",
            "source_dir": scripts_dir,
            "image_uri": self.config.image_uri,
            "role": self.config.role,
            "instance_count": self.config.instance_count,
            "instance_type": self.config.instance_type,
            "sagemaker_session": sagemaker_session,
            "output_path": output_path,
            "code_location": code_location,
            "base_job_name": self.config.base_job_name,
            "hyperparameters": job_config.to_hyperparameters(),
            "max_run": self.config.max_runtime_seconds,
            "disable_profiler": True,
            "volume_size": 100,
        }

        # Add environment variables if any
        env_vars = job_config.to_environment()
        if env_vars:
            estimator_kwargs["environment"] = env_vars

        # Add VPC configuration if provided (required for classified networks)
        if self.config.subnets:
            estimator_kwargs["subnets"] = self.config.subnets
        if self.config.security_group_ids:
            estimator_kwargs["security_group_ids"] = self.config.security_group_ids

        # Add tags if provided
        if self.config.tags:
            estimator_kwargs["tags"] = [
                {"Key": k, "Value": v} for k, v in self.config.tags.items()
            ]

        # Add network isolation if enabled
        if self.config.enable_network_isolation:
            estimator_kwargs["enable_network_isolation"] = True

        logger.info(f"Creating PyTorch Estimator for job: {job_name}")
        logger.debug(f"Estimator kwargs: {json.dumps({k: str(v) for k, v in estimator_kwargs.items()}, indent=2)}")

        # Create estimator
        estimator = PyTorch(**estimator_kwargs)

        # Start training job
        logger.info(f"Starting training job: {job_name}")
        estimator.fit(
            inputs={"training": job_config.dataset_s3_uri},
            job_name=job_name,
            wait=wait,
            logs=wait,  # Show logs if waiting
        )

        logger.info(f"Training job started: {job_name}")

        return job_name

    def get_job_status(self, job_name: str) -> Dict[str, Any]:
        """Get the status of a training job.

        Args:
            job_name: Training job name.

        Returns:
            Dict with job status information including serialized datetimes.

        Raises:
            ClientError: If job not found or API error.
        """
        response = self.sagemaker_client.describe_training_job(
            TrainingJobName=job_name
        )

        status = {
            "job_name": job_name,
            "status": response["TrainingJobStatus"],
            "secondary_status": response.get("SecondaryStatus"),
            "creation_time": _serialize_datetime(response.get("CreationTime")),
            "training_start_time": _serialize_datetime(response.get("TrainingStartTime")),
            "training_end_time": _serialize_datetime(response.get("TrainingEndTime")),
            "failure_reason": response.get("FailureReason"),
            "model_artifacts": response.get("ModelArtifacts", {}).get("S3ModelArtifacts"),
            "billable_seconds": response.get("BillableTimeInSeconds"),
        }

        # Add metrics if available
        if "FinalMetricDataList" in response:
            status["metrics"] = {
                m["MetricName"]: m["Value"]
                for m in response["FinalMetricDataList"]
            }

        # Include hyperparameters for panel display (e.g., W&B project info)
        if "HyperParameters" in response:
            status["hyperparameters"] = response["HyperParameters"]

        # Include environment variables for W&B link construction
        if "Environment" in response:
            status["environment"] = response["Environment"]

        return status

    def wait_for_job(
        self,
        job_name: str,
        poll_interval: int = 30,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Wait for a training job to complete.

        Args:
            job_name: Training job name.
            poll_interval: Seconds between status checks.
            timeout: Maximum seconds to wait (None for no timeout).

        Returns:
            Final job status.

        Raises:
            TimeoutError: If timeout exceeded.
            RuntimeError: If job failed.
        """
        logger.info(f"Waiting for training job: {job_name}")

        start_time = time.time()
        terminal_states = {"Completed", "Failed", "Stopped"}

        while True:
            status = self.get_job_status(job_name)
            current_status = status["status"]
            secondary = status.get("secondary_status", "")

            logger.info(f"Job {job_name}: {current_status} ({secondary})")

            if current_status in terminal_states:
                if current_status == "Failed":
                    raise RuntimeError(
                        f"Training job failed: {status.get('failure_reason', 'Unknown reason')}"
                    )
                return status

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Timeout waiting for job {job_name}")

            time.sleep(poll_interval)

    def list_jobs(
        self,
        name_contains: Optional[str] = None,
        status_equals: Optional[str] = None,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """List training jobs.

        Args:
            name_contains: Filter by job name substring.
            status_equals: Filter by status (InProgress, Completed, Failed, etc.).
            max_results: Maximum number of results.

        Returns:
            List of job summaries with serialized datetimes.
        """
        kwargs = {
            "MaxResults": max_results,
            "SortBy": "CreationTime",
            "SortOrder": "Descending",
        }

        if name_contains:
            kwargs["NameContains"] = name_contains

        if status_equals:
            kwargs["StatusEquals"] = status_equals

        response = self.sagemaker_client.list_training_jobs(**kwargs)

        jobs = []
        for job in response.get("TrainingJobSummaries", []):
            jobs.append({
                "job_name": job["TrainingJobName"],
                "status": job["TrainingJobStatus"],
                "creation_time": _serialize_datetime(job["CreationTime"]),
                "training_end_time": _serialize_datetime(job.get("TrainingEndTime")),
            })

        return jobs

    def download_artifacts(
        self,
        job_name: str,
        local_path: Path,
        extract: bool = True,
    ) -> Path:
        """Download model artifacts from a completed training job.

        Args:
            job_name: Training job name.
            local_path: Local directory to download to.
            extract: Whether to extract the model.tar.gz archive.

        Returns:
            Path to downloaded artifacts.

        Raises:
            RuntimeError: If job not completed or artifacts not available.
        """
        status = self.get_job_status(job_name)

        if status["status"] != "Completed":
            raise RuntimeError(
                f"Cannot download artifacts: job status is {status['status']}"
            )

        model_artifacts_uri = status.get("model_artifacts")
        if not model_artifacts_uri:
            raise RuntimeError("No model artifacts available")

        local_path = Path(local_path)
        local_path.mkdir(parents=True, exist_ok=True)

        # Parse S3 URI
        # Format: s3://bucket/key/model.tar.gz
        parts = model_artifacts_uri.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        key = parts[1]

        # Download the tarball
        tarball_path = local_path / "model.tar.gz"
        logger.info(f"Downloading {model_artifacts_uri} to {tarball_path}")

        self.s3_client.download_file(bucket, key, str(tarball_path))

        if extract:
            logger.info(f"Extracting artifacts to {local_path}")
            with tarfile.open(tarball_path, "r:gz") as tar:
                # Filter to prevent path traversal (CVE-2007-4559)
                tar.extractall(local_path, filter="data")
            tarball_path.unlink()  # Remove tarball after extraction

        logger.info(f"Artifacts downloaded to {local_path}")
        return local_path

    def stop_job(self, job_name: str) -> None:
        """Stop a running training job.

        Args:
            job_name: Training job name.
        """
        logger.info(f"Stopping training job: {job_name}")
        self.sagemaker_client.stop_training_job(TrainingJobName=job_name)
        logger.info(f"Stop request sent for job: {job_name}")


