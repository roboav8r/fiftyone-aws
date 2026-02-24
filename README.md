# FiftyOne AWS SageMaker Plugin

A [FiftyOne](https://docs.voxel51.com/) plugin for training YOLO models on AWS SageMaker. Supports custom ECR images, VPC isolation, and classified network deployments.

## Features

- **SageMaker Training**: Launch YOLO26 training jobs directly from the FiftyOne App
- **Custom ECR Support**: Use your own Docker images (required for classified/GovCloud networks)
- **VPC Configuration**: Subnet and security group support for network-isolated environments
- **Job Monitoring**: List, inspect, and stop training jobs from the UI
- **Model Inference**: Apply trained models back to your FiftyOne dataset
- **Flexible Data Splits**: Split by tags, saved views, or random ratios
- **W&B Integration**: Optional Weights & Biases experiment tracking
- **Python Panel**: Full-featured panel UI with Train, Monitor, and Apply tabs

## Operators

| Operator | Description |
|---|---|
| `launch_sagemaker_training` | Export dataset, upload to S3, and launch a SageMaker training job |
| `get_training_job_status` | Check status of a SageMaker training job |
| `list_training_jobs` | List recent SageMaker training jobs |
| `download_model_artifacts` | Download trained model weights from a completed job |
| `stop_training_job` | Stop a running training job |
| `apply_split_tags` | Apply random train/val/test split tags to samples |
| `apply_yolo_model` | Run inference with a trained YOLO model on the current view |

## Panel

The **SageMaker Trainer** panel provides a tabbed interface:

- **Train**: Configure AWS, instance, training hyperparameters, and data splits. Launch jobs with one click.
- **Monitor**: List recent jobs, view detailed status, metrics, and stop running jobs.
- **Apply Model**: Load trained weights (from local path or completed job) and run inference on your dataset.

## Installation

### FiftyOne Open-Source

```shell
fiftyone plugins download https://github.com/roboav8r/fiftyone-aws
```

### FiftyOne Teams (Enterprise)

For FiftyOne Teams deployments, the plugin must be installed into the application server's Docker image and registered via environment variables.

#### Docker Image

Create a custom Dockerfile that extends the FiftyOne Teams App image:

```dockerfile
FROM voxel51/fiftyone-teams-app:latest

# Install plugin dependencies
# Note: sagemaker has a protobuf dependency that can conflict with fiftyone.
# Install with --no-deps and add only the needed sub-dependencies.
RUN pip install \
    boto3>=1.28.0 \
    pyyaml>=6.0 \
    ultralytics>=8.3.0 && \
    pip install --no-deps sagemaker==2.254.1 && \
    pip install schema docker pathos

# Copy plugin into the container
COPY . /opt/plugins/fiftyone-aws/
```

Build and push:

```shell
docker build -t my-registry/fiftyone-teams-app:latest .
docker push my-registry/fiftyone-teams-app:latest
```

#### Docker Compose

Add the plugin directory and environment variables to your `docker-compose.yml`:

```yaml
services:
  fiftyone-app:
    image: my-registry/fiftyone-teams-app:latest
    environment:
      - FIFTYONE_PLUGINS_DIR=/opt/plugins
```

#### Kubernetes (Helm)

In your Helm values file:

```yaml
appSettings:
  env:
    FIFTYONE_PLUGINS_DIR: /opt/plugins
```

## Secrets Configuration

This plugin uses three secrets for credentials. **Never** store these in config files.

| Secret | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | AWS access key for SageMaker and S3 |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key |
| `WANDB_API_KEY` | Weights & Biases API key (optional) |

### FiftyOne Teams (Secrets Manager)

Configure secrets in the FiftyOne Teams admin UI under **Settings > Secrets**. These are encrypted and stored in the database.

### FiftyOne Open-Source (Environment Variables)

Set secrets as environment variables on the machine running the FiftyOne App server:

```shell
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export WANDB_API_KEY=...  # optional
```

## Default Configuration

Copy `default_config.example.yaml` to `default_config.yaml` in the plugin directory and fill in your site-specific defaults. These values pre-populate the operator and panel UI fields.

```shell
cp default_config.example.yaml default_config.yaml
```

The config file contains **non-sensitive** defaults only (region, instance type, model, bucket, etc.). Credentials are always resolved via secrets.

## Usage

### Operator Palette

Press `` ` `` in the FiftyOne App to open the operator palette, then search for any of the operators listed above.

### Panel

1. Open the FiftyOne App
2. Click the **+** button in the panel area and select **SageMaker Trainer**
3. Use the **Train** tab to configure and launch a training job
4. Switch to **Monitor** to track job progress
5. Use **Apply Model** to run inference with trained weights

### Typical Workflow

1. Load your dataset in FiftyOne
2. Apply split tags (Train tab > "Apply Split Tags" button) or use saved views
3. Configure AWS settings (ECR image, role, bucket)
4. Set training hyperparameters (model, epochs, batch size, etc.)
5. Click "Launch Training Job"
6. Monitor progress in the Monitor tab
7. Once complete, apply the trained model in the Apply tab

## AWS Setup

### IAM Permissions

The SageMaker execution role needs:

- `sagemaker:CreateTrainingJob`, `sagemaker:DescribeTrainingJob`, `sagemaker:ListTrainingJobs`, `sagemaker:StopTrainingJob`
- `s3:GetObject`, `s3:PutObject`, `s3:ListBucket` on your training bucket
- `ecr:GetAuthorizationToken`, `ecr:BatchGetImage` for pulling training images
- `logs:CreateLogGroup`, `logs:CreateLogStream`, `logs:PutLogEvents` for CloudWatch logs

### S3 Bucket

Create an S3 bucket for training data and model artifacts. The plugin uploads datasets to `s3://{bucket}/datasets/` and reads outputs from `s3://{bucket}/output/`.

### ECR Image

The plugin requires a custom ECR image URI. This is the Docker image SageMaker uses to run the training job. Use a PyTorch training image that includes GPU support:

```
{account_id}.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker
```

## Classified Network Considerations

This plugin was designed for use on classified AWS networks (GovCloud, IC regions):

- **Custom ECR**: The image URI field is required (not auto-resolved) because classified network ECR endpoints differ from commercial AWS
- **VPC Isolation**: Configure subnets and security groups to run training jobs within your VPC
- **Air-Gapped W&B**: If using W&B behind a firewall, set `WANDB_BASE_URL` as an environment variable in your training container
- **No Internet Access**: The training script and all dependencies are bundled in the ECR image and uploaded source code — no runtime downloads required

## Troubleshooting

### Training job fails immediately

- Verify the ECR image URI is accessible from SageMaker
- Check the IAM role has permissions to pull from ECR and access S3
- If using VPC subnets, ensure they have a NAT gateway or VPC endpoints for S3 and ECR

### "No samples found with tag" error

- Apply split tags first using the "Apply Split Tags" button in the Train tab
- Verify your dataset has samples with the expected tags (`train`, `val`, `test`)

### Plugin not appearing in FiftyOne App

- Verify `FIFTYONE_PLUGINS_DIR` is set and points to the parent directory containing the plugin
- Check that `fiftyone.yml` is present in the plugin directory
- Restart the FiftyOne App server after installing the plugin

### SageMaker SDK protobuf conflict

When installing in a FiftyOne Teams Docker image, install `sagemaker` with `--no-deps` to avoid protobuf version conflicts:

```shell
pip install --no-deps sagemaker==2.254.1
pip install schema docker pathos
```

## License

Apache-2.0. See [LICENSE](LICENSE).
