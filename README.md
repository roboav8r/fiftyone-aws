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

For FiftyOne Teams deployments, the plugin dependencies must be installed into a custom Docker image. The custom image is then used for the `teams-plugins` and `teams-do` services via a compose override file.

See the [FiftyOne Teams Docker deployment guide](https://github.com/voxel51/fiftyone-teams-app-deploy/blob/main/docker/README.md) for full reference.

#### 1. Build a Custom Docker Image

Create a `Dockerfile` that extends your base FiftyOne Teams image with the plugin's dependencies:

```dockerfile
ARG BASE_IMAGE

FROM ${BASE_IMAGE}

# Install dependencies
RUN pip install --no-cache-dir \
    boto3 \
    python-dotenv \
    pyyaml \
    wandb

# Remove any existing sagemaker packages that might conflict
RUN pip uninstall -y sagemaker sagemaker-core || true

# Clean up any leftover sagemaker directories
RUN rm -rf /opt/fiftyone-teams-app/lib/python3.11/site-packages/sagemaker* || true

# Now install fresh
RUN pip install --no-cache-dir sagemaker==2.254.1

# Verify import
RUN python -c "import sagemaker; print('sagemaker', sagemaker.__version__)"
```

Build the image, passing your base FiftyOne Teams image as a build arg:

```shell
docker build \
    --no-cache \
    --build-arg BASE_IMAGE="voxel51/fiftyone-teams-cv-full:v2.16.0" \
    -t my-fiftyone-sagemaker:v2.16.0 \
    .
```

#### 2. Configure Docker Compose Override

In your FiftyOne Teams deployment directory, update `compose.override.yaml` to use the custom image for the `teams-plugins` and `teams-do` services (which execute plugin code and delegated operators):

```yaml
services:
  teams-plugins:
    image: my-fiftyone-sagemaker:v2.16.0
    pull_policy: never  # if using a locally-built image

  teams-do:
    image: my-fiftyone-sagemaker:v2.16.0
    pull_policy: never  # if using a locally-built image
```

> **Note:** The `fiftyone-app` service does not need the custom image -- only `teams-plugins` (runs plugin UI code) and `teams-do` (runs delegated operators) need the SageMaker dependencies.

#### 3. Restart Services

Bring the stack down and back up with all compose files:

```shell
docker compose down

docker compose \
    -f compose.yaml \
    -f compose.delegated-operators.yaml \
    -f compose.dedicated-plugins.yaml \
    -f compose.override.yaml \
    up -d
```

#### 4. Install the Plugin

Once the services are running, install the plugin from GitHub using the FiftyOne CLI or the App's plugin management UI:

```shell
fiftyone plugins download https://github.com/roboav8r/fiftyone-aws
```

#### Kubernetes (Helm)

For Helm-based deployments, build and push the custom image to a registry accessible from your cluster, then configure `values.yaml` to use it for the dedicated plugins and delegated operator deployments.

See the [FiftyOne Teams Helm deployment guide](https://github.com/voxel51/fiftyone-teams-app-deploy/tree/main/helm) for full reference, including [plugin configuration](https://github.com/voxel51/fiftyone-teams-app-deploy/blob/main/helm/docs/configuring-plugins.md) and [delegated operator configuration](https://github.com/voxel51/fiftyone-teams-app-deploy/blob/main/helm/docs/configuring-delegated-operators.md).

**1. Build and push the custom image** (same Dockerfile as Docker Compose):

```shell
docker build \
    --no-cache \
    --build-arg BASE_IMAGE="voxel51/fiftyone-teams-cv-full:v2.16.0" \
    -t my-registry/fiftyone-sagemaker:v2.16.0 \
    .

docker push my-registry/fiftyone-sagemaker:v2.16.0
```

**2. Configure dedicated plugins** in `values.yaml`:

```yaml
pluginsSettings:
  enabled: true
  image:
    repository: my-registry/fiftyone-sagemaker
    tag: v2.16.0
  env:
    FIFTYONE_PLUGINS_DIR: /opt/plugins

apiSettings:
  env:
    FIFTYONE_PLUGINS_DIR: /opt/plugins
```

Mount a PersistentVolumeClaim with `ReadWrite` access to `teams-api` and `ReadOnly` access to `teams-plugins` at the `FIFTYONE_PLUGINS_DIR` path. See [plugins-storage.md](https://github.com/voxel51/fiftyone-teams-app-deploy/blob/main/helm/docs/plugins-storage.md) for PVC configuration.

**3. Configure delegated operators** in `values.yaml`.

For always-on executors (`delegatedOperatorDeployments`):

```yaml
delegatedOperatorDeployments:
  deployments:
    teamsDo:
      image:
        repository: my-registry/fiftyone-sagemaker
        tag: v2.16.0
      env:
        FIFTYONE_PLUGINS_DIR: /opt/plugins
      volumes:
        - name: plugins-vol
          persistentVolumeClaim:
            claimName: plugins-pvc
            readOnly: true
      volumeMounts:
        - name: plugins-vol
          mountPath: /opt/plugins
```

For on-demand executors (`delegatedOperatorJobTemplates`):

```yaml
delegatedOperatorJobTemplates:
  template:
    image:
      repository: my-registry/fiftyone-sagemaker
      tag: v2.16.0
    env:
      FIFTYONE_PLUGINS_DIR: /opt/plugins
    volumes:
      - name: plugins-vol
        persistentVolumeClaim:
          claimName: plugins-pvc
          readOnly: true
    volumeMounts:
      - name: plugins-vol
        mountPath: /opt/plugins
  jobs:
    teamsDoCpuDefaultK8s: {}
```

**4. Deploy or upgrade:**

```shell
helm repo add voxel51 https://helm.fiftyone.ai
helm repo update voxel51

# New install
helm install fiftyone-teams-app voxel51/fiftyone-teams-app -f values.yaml

# Or upgrade existing
helm upgrade fiftyone-teams-app voxel51/fiftyone-teams-app -f values.yaml
```

**5. Install the plugin** once the pods are running:

```shell
fiftyone plugins download https://github.com/roboav8r/fiftyone-aws
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

### SageMaker SDK conflicts in Docker image

The `sagemaker` package can conflict with packages already installed in the FiftyOne Teams base image. If you see import errors, ensure your Dockerfile removes existing sagemaker packages before installing fresh:

```dockerfile
RUN pip uninstall -y sagemaker sagemaker-core || true
RUN rm -rf /opt/fiftyone-teams-app/lib/python3.11/site-packages/sagemaker* || true
RUN pip install --no-cache-dir sagemaker==2.254.1
```

## License

Apache-2.0. See [LICENSE](LICENSE).
