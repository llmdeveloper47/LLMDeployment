# Deploying and Optimizing LLMs on Google Kubernetes Engine: End-to-End MLOps Guide

This comprehensive guide provides detailed, step-by-step instructions for deploying, optimizing, and managing Large Language Models (LLMs) on Google Kubernetes Engine (GKE). The solution incorporates MLOps best practices including infrastructure as code, containerization, CI/CD, blue-green deployments, model quantization, and performance benchmarking.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Google Cloud Setup](#google-cloud-setup)
4. [Infrastructure Setup with Terraform](#infrastructure-setup-with-terraform)
5. [Building and Pushing the Docker Image](#building-and-pushing-the-docker-image)
6. [Deploying the LLM Service](#deploying-the-llm-service)
7. [Setting Up Streaming Inference](#setting-up-streaming-inference)
8. [Load Testing](#load-testing)
9. [Model Quantization for Performance Optimization](#model-quantization-for-performance-optimization)
10. [Blue-Green Deployment Strategy](#blue-green-deployment-strategy)
11. [CI/CD Setup](#cicd-setup)
12. [Monitoring and Scaling](#monitoring-and-scaling)
13. [Cost Management](#cost-management)
14. [Cleanup Process](#cleanup-process)
15. [Troubleshooting](#troubleshooting)

## Overview

This solution provides a complete MLOps pipeline for LLMs on GKE with the following features:

- **Infrastructure as Code**: Terraform scripts for GKE cluster provisioning
- **Containerized Inference**: Optimized Docker image for LLM serving
- **Streaming Inference**: FastAPI application with streaming response capabilities
- **Blue-Green Deployments**: Safe model updates with zero downtime
- **Horizontal Scaling**: Kubernetes HPA for auto-scaling based on load
- **Model Compression**: Quantization to reduce size and improve performance
- **CI/CD Pipeline**: GitHub Actions workflow for automated deployments
- **Load Testing**: Performance benchmarking with Locust
- **Cleanup Utilities**: Terraform destroy scripts to minimize costs

## Prerequisites

Before you begin, ensure you have the following tools installed:

1. **Google Cloud SDK**: For interacting with Google Cloud services
2. **kubectl**: For managing Kubernetes resources
3. **Terraform**: For infrastructure as code (v1.0.0+)
4. **Docker**: For building and testing container images
5. **Python 3.8+**: For running scripts and utilities
6. **Git**: For version control and CI/CD setup

## Google Cloud Setup

### 1. Create a Google Cloud Account

1. Visit [cloud.google.com](https://cloud.google.com)
2. Click "Get started for free" or "Sign in"
3. Follow the prompts to create a new account or sign in
4. Set up billing information (you may be eligible for free credits)

### 2. Create a New GCP Project

```bash
# Create a new project
gcloud projects create llm-deployment-project --name="LLM Deployment"

# Set the project as your default
gcloud config set project llm-deployment-project

# Verify project ID
gcloud config get-value project
```

### 3. Enable Required APIs

```bash
# Enable required Google Cloud APIs
gcloud services enable compute.googleapis.com
gcloud services enable container.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable storage-component.googleapis.com
gcloud services enable storage-api.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable iam.googleapis.com
gcloud services enable cloudresourcemanager.googleapis.com
```

### 4. Create a Service Account for Terraform

```bash
# Create a service account for Terraform
gcloud iam service-accounts create terraform-admin \
    --description="Terraform Admin Account" \
    --display-name="Terraform Admin"

# Grant necessary roles to the service account
gcloud projects add-iam-policy-binding llm-deployment-project \
    --member="serviceAccount:terraform-admin@llm-deployment-project.iam.gserviceaccount.com" \
    --role="roles/compute.admin"

gcloud projects add-iam-policy-binding llm-deployment-project \
    --member="serviceAccount:terraform-admin@llm-deployment-project.iam.gserviceaccount.com" \
    --role="roles/container.admin"

gcloud projects add-iam-policy-binding llm-deployment-project \
    --member="serviceAccount:terraform-admin@llm-deployment-project.iam.gserviceaccount.com" \
    --role="roles/storage.admin"

gcloud projects add-iam-policy-binding llm-deployment-project \
    --member="serviceAccount:terraform-admin@llm-deployment-project.iam.gserviceaccount.com" \
    --role="roles/iam.serviceAccountUser"

gcloud projects add-iam-policy-binding llm-deployment-project \
    --member="serviceAccount:terraform-admin@llm-deployment-project.iam.gserviceaccount.com" \
    --role="roles/resourcemanager.projectIamAdmin"

# Create and download a key for the service account
gcloud iam service-accounts keys create terraform-admin-key.json \
    --iam-account=terraform-admin@llm-deployment-project.iam.gserviceaccount.com
```

Make sure to store this key file securely. You'll use it for Terraform authentication.

## Infrastructure Setup with Terraform

### 1. Clone the Repository

```bash
# Clone your repository (replace with your actual repository URL)
git clone https://github.com/yourusername/llm-gke-deployment.git
cd llm-gke-deployment
```

### 2. Configure Terraform Variables

Create or edit `terraform/terraform.tfvars` with your specific values:

```hcl
project_id     = "llm-deployment-project"
region         = "us-central1"
gke_num_nodes  = 2
machine_type   = "n2-standard-8"
# For GPU support (uncomment as needed)
# gpu_type     = "nvidia-tesla-t4"
# gpu_count    = 1
```

### 3. Initialize Terraform

```bash
cd terraform

# Set environment variable for authentication
export GOOGLE_APPLICATION_CREDENTIALS="../terraform-admin-key.json"

# Initialize Terraform
terraform init
```

### 4. Create Execution Plan

```bash
terraform plan -out=tfplan
```

Review the plan to confirm the resources that will be created.

### 5. Apply the Plan

```bash
terraform apply tfplan
```

This will create:
- A VPC network with subnets
- A GKE cluster with the specified node configuration
- A Cloud Storage bucket for model storage
- Required service accounts and permissions

The process takes approximately 10-15 minutes.

### 6. Configure kubectl

```bash
# Get cluster credentials and configure kubectl
gcloud container clusters get-credentials $(terraform output -raw kubernetes_cluster_name) \
    --region $(terraform output -raw region) \
    --project $(terraform output -raw project_id)
```

## Building and Pushing the Docker Image

### 1. Prepare the Dockerfile and Application Code

Ensure you have the following files in your project:
- `Dockerfile`: Container definition optimized for LLM inference
- `requirements.txt`: Python dependencies
- `app/main.py`: FastAPI application for inference
- `scripts/download_model.py`: Utility to download models

### 2. Build the Docker Image

```bash
# Set environment variables
export PROJECT_ID=$(gcloud config get-value project)
export REGION=$(gcloud config get-value compute/region)
export REGISTRY="gcr.io/${PROJECT_ID}"
export IMAGE="llm-inference"
export TAG="v1"

# Build the image
docker build -t ${REGISTRY}/${IMAGE}:${TAG} .
```

### 3. Test the Image Locally (Optional)

```bash
# Run the container locally
docker run -p 8000:8000 \
    -e MODEL_ID="meta-llama/Llama-2-7b-chat-hf" \
    -e DEFAULT_MAX_LENGTH="1024" \
    -e USE_4BIT="true" \
    ${REGISTRY}/${IMAGE}:${TAG}
```

### 4. Push the Image to Google Container Registry

```bash
# Configure Docker to use gcloud credentials
gcloud auth configure-docker

# Push the image
docker push ${REGISTRY}/${IMAGE}:${TAG}
```

## Deploying the LLM Service

### 1. Create Kubernetes ConfigMap and Secret

```bash
# Update configmap.yaml with your settings
cat > k8s/configmap.yaml << EOL
apiVersion: v1
kind: ConfigMap
metadata:
  name: llm-inference-config
data:
  MODEL_ID: "meta-llama/Llama-2-7b-chat-hf"
  DEFAULT_MAX_LENGTH: "1024"
  DEFAULT_TEMPERATURE: "0.7"
  USE_4BIT: "true"
  STREAMING_MODE: "true"
---
apiVersion: v1
kind: Secret
metadata:
  name: hf-token
type: Opaque
stringData:
  token: "your-huggingface-token-here"  # Replace with your actual token
EOL

# Apply the configmap and secret
kubectl apply -f k8s/configmap.yaml
```

### 2. Update Deployment Configuration

Edit `k8s/deployment.yaml` to use your specific values:

```bash
# Replace placeholder values in deployment.yaml
sed -i "s|\${CONTAINER_REGISTRY}|${REGISTRY}|g" k8s/deployment.yaml
sed -i "s|\${TAG}|${TAG}|g" k8s/deployment.yaml
sed -i "s|\${MODEL_ID}|meta-llama/Llama-2-7b-chat-hf|g" k8s/deployment.yaml
```

### 3. Deploy to Kubernetes

```bash
# Apply the deployment
kubectl apply -f k8s/deployment.yaml

# Apply the service and other resources
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Check status
kubectl get pods
kubectl get deployments
kubectl get services

# Wait for the deployment to be ready
kubectl rollout status deployment/llm-inference --timeout=15m
```

This may take several minutes as the model is downloaded during initialization.

## Setting Up Streaming Inference

The deployment already includes a FastAPI application that supports streaming inference. To test it:

### 1. Get the Service Endpoint

```bash
# Get the service external IP or set up port forwarding
EXTERNAL_IP=$(kubectl get svc llm-inference -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# If using port forwarding instead
if [ -z "$EXTERNAL_IP" ]; then
  kubectl port-forward svc/llm-inference 8000:80 &
  ENDPOINT="http://localhost:8000"
else
  ENDPOINT="http://${EXTERNAL_IP}"
fi
```

### 2. Test Non-Streaming Inference

```bash
# Test with curl
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing in simple terms.",
    "max_length": 150,
    "temperature": 0.7,
    "stream": false
  }' \
  ${ENDPOINT}/generate
```

### 3. Test Streaming Inference

For streaming inference, you'll need a client that supports Server-Sent Events (SSE). You can use our Streamlit demo app:

```bash
# Install Streamlit and dependencies
pip install streamlit plotly pandas requests sseclient-py

# Run the Streamlit app
API_ENDPOINT=${ENDPOINT} streamlit run demo/app.py
```

Visit http://localhost:8501 to interact with the LLM through the UI.

## Load Testing

### 1. Install Locust

```bash
pip install locust sseclient-py
```

### 2. Run Load Tests

```bash
# Run Locust with the web UI
ENDPOINT=${ENDPOINT} locust -f scripts/load_test.py
```

Open a browser at http://localhost:8089 and configure:
- Number of users: Start with 10
- Spawn rate: 1 user per second
- Host: Your service endpoint

### 3. Analyze Results

Locust provides real-time graphs and statistics. Pay attention to:
- Response time
- Requests per second
- Failure rate

For a more detailed analysis, you can use our benchmark script:

```bash
# Run benchmarks
python scripts/benchmark_model.py \
  --endpoint-a ${ENDPOINT} \
  --output-dir ./benchmark_results \
  --max-length 150
```

## Model Quantization for Performance Optimization

### 1. Download the Base Model

```bash
# Create directories for model files
mkdir -p tmp_model quantized_model

# Download the model
python scripts/download_model.py \
  --model-id "meta-llama/Llama-2-7b-chat-hf" \
  --output-path ./tmp_model
```

### 2. Quantize the Model

```bash
# Quantize to 4-bit precision using BitsAndBytes
python scripts/quantize_model.py \
  --input-model ./tmp_model \
  --output-model ./quantized_model \
  --method bnb \
  --bits 4
```

### 3. Upload to Google Cloud Storage

```bash
# Set variables
MODEL_NAME="llama-2-7b-chat-bnb4bit"
BUCKET_NAME=$(terraform -chdir=terraform output -raw model_storage_bucket)

# Upload to GCS
gsutil -m cp -r ./quantized_model/* gs://${BUCKET_NAME}/${MODEL_NAME}/
```

## Blue-Green Deployment Strategy

### 1. Update Kubernetes Manifests

First, ensure you've switched to using the blue-green deployment configuration:

```bash
# Apply the blue-green deployment configuration
kubectl apply -f k8s/blue-green-deployment.yaml
```

### 2. Prepare for Blue-Green Deployment

Use our `update_model.py` script to manage the blue-green deployment process:

```bash
# Run the model update script
python scripts/update_model.py \
  --model-path "gs://${BUCKET_NAME}/${MODEL_NAME}" \
  --skip-tests false
```

This script will:
1. Create a new "green" deployment with the quantized model
2. Set up a separate service for testing
3. Run validation tests
4. Prompt for traffic switching

### 3. Compare Performance Before Switching

Before fully committing to the quantized model, benchmark it against the original:

```bash
# Get the green service endpoint
GREEN_ENDPOINT="http://$(kubectl get svc llm-inference-green-svc -o jsonpath='{.status.loadBalancer.ingress[0].ip}')"

# If using port forwarding
if [[ -z "$GREEN_ENDPOINT" || "$GREEN_ENDPOINT" == "http://" ]]; then
  kubectl port-forward svc/llm-inference-green-svc 8001:80 &
  sleep 5
  GREEN_ENDPOINT="http://localhost:8001"
fi

# Run benchmarks comparing both models
python scripts/benchmark_model.py \
  --endpoint-a ${GREEN_ENDPOINT} \
  --endpoint-b ${ENDPOINT} \
  --output-dir "./benchmark_results/${MODEL_NAME}" \
  --max-length 150
```

### 4. Switch Traffic to Quantized Model

If the benchmarks are satisfactory, switch traffic to the new deployment:

```bash
# Switch traffic to the green deployment
kubectl patch service llm-inference -p '{"spec":{"selector":{"version":"green"}}}'

# Scale up the green deployment
kubectl scale deployment/llm-inference-green --replicas=2

# Optionally scale down the blue deployment
kubectl scale deployment/llm-inference --replicas=0
```

### 5. Rollback (If Needed)

If issues are detected, you can quickly roll back:

```bash
# Switch traffic back to the blue deployment
kubectl patch service llm-inference -p '{"spec":{"selector":{"version":"blue"}}}'

# Scale the blue deployment back up
kubectl scale deployment/llm-inference --replicas=2

# Scale down the green deployment
kubectl scale deployment/llm-inference-green --replicas=0
```

### 6. Automated Blue-Green Deployment

For convenience, you can use our script that automates the entire process:

```bash
# Make the script executable
chmod +x scripts/quantize_and_deploy.sh

# Run with default settings
./scripts/quantize_and_deploy.sh \
  --model-id "meta-llama/Llama-2-7b-chat-hf" \
  --quantize-method bnb \
  --bits 4 \
  --auto-switch  # Optional: automatically switch traffic if tests pass
```

## CI/CD Setup

### 1. Prepare GitHub Repository

Ensure your code is in a GitHub repository with the following files:
- `.github/workflows/deploy-model.yaml`: CI/CD workflow configuration
- All application code and infrastructure files

### 2. Set Up GitHub Secrets

In your GitHub repository:
1. Go to Settings > Secrets and variables > Actions
2. Add the following secrets:
   - `GCP_PROJECT_ID`: Your Google Cloud project ID
   - `GCP_SA_KEY`: Base64-encoded service account key (create a dedicated CI/CD service account)
   - `GKE_CLUSTER_NAME`: Your GKE cluster name
   - `GKE_ZONE`: Your GKE cluster zone/region
   - `GCP_REGISTRY`: Your container registry (e.g., gcr.io/your-project-id)
   - `GCS_BUCKET`: Your GCS bucket for model storage
   - `HF_TOKEN`: Your Hugging Face token (if using private models)

To base64 encode your service account key:

```bash
cat service-account-key.json | base64 -w 0
```

### 3. Create a CI/CD Service Account

```bash
# Create service account for GitHub Actions
gcloud iam service-accounts create github-actions \
    --description="GitHub Actions CI/CD Account" \
    --display-name="GitHub Actions"

# Grant necessary permissions
gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
    --member="serviceAccount:github-actions@$(gcloud config get-value project).iam.gserviceaccount.com" \
    --role="roles/container.developer"

gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
    --member="serviceAccount:github-actions@$(gcloud config get-value project).iam.gserviceaccount.com" \
    --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
    --member="serviceAccount:github-actions@$(gcloud config get-value project).iam.gserviceaccount.com" \
    --role="roles/compute.admin"

# Create and download key
gcloud iam service-accounts keys create github-actions-key.json \
    --iam-account=github-actions@$(gcloud config get-value project).iam.gserviceaccount.com
```

Use this key to create the `GCP_SA_KEY` GitHub secret.

### 4. Triggering CI/CD Workflow

The workflow will automatically trigger on:
- Pushes to the main branch
- Manual workflow dispatch (for specific model deployments)

You can also trigger it manually from the GitHub Actions tab in your repository.

## Monitoring and Scaling

### 1. Configure HPA for Automatic Scaling

The HPA is already applied with default settings. You can customize it:

```bash
# Edit the HPA configuration
kubectl edit hpa llm-inference-hpa
```

Key settings to consider:
- `minReplicas`: Minimum number of pods (default: 1)
- `maxReplicas`: Maximum number of pods (default: 10)
- Target CPU and memory utilization percentages

### 2. Set Up Google Cloud Monitoring

```bash
# Install the Kubernetes Monitoring agent
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/prometheus-engine/main/manifests/setup.yaml

# Create a sample dashboard in Cloud Monitoring
gcloud monitoring dashboards create \
  --config-from-file=monitoring/dashboard.json
```

### 3. Set Up Alerts

```bash
# Create alerts for high error rates
gcloud alpha monitoring policies create \
  --policy-from-file=monitoring/alerts/error_rate_alert.json

# Create alerts for high latency
gcloud alpha monitoring policies create \
  --policy-from-file=monitoring/alerts/latency_alert.json
```

## Cost Management

### 1. Resource Requests and Limits

Review and adjust resource requests and limits in `k8s/deployment.yaml` to optimize costs:

```yaml
resources:
  requests:
    cpu: "2"
    memory: "8Gi"
  limits:
    cpu: "4"
    memory: "16Gi"
```

### 2. Use Preemptible VMs for Development

For non-production environments, consider using preemptible VMs:

```bash
# Update node pool to use preemptible VMs
gcloud container node-pools update primary-nodes \
    --cluster=$(terraform output -raw kubernetes_cluster_name) \
    --region=$(terraform output -raw region) \
    --preemptible
```

### 3. Scale Down During Off-Hours

For development or non-24/7 workloads:

```bash
# Scale down deployments during off-hours
kubectl scale deployment/llm-inference --replicas=0

# Scale back up when needed
kubectl scale deployment/llm-inference --replicas=2
```

## Cleanup Process

When you're done testing and want to avoid incurring costs:

```bash
# Make the cleanup script executable
chmod +x scripts/cleanup.sh

# Run the cleanup script
./scripts/cleanup.sh
```

The script will:
1. Scale down deployments to 0
2. Delete Kubernetes resources
3. Run Terraform destroy to remove all infrastructure
4. Delete the GCS bucket and contents
5. Delete Docker images from the container registry

Alternatively, you can manually clean up:

```bash
# 1. Scale down deployments
kubectl scale deployment llm-inference --replicas=0
kubectl scale deployment llm-inference-green --replicas=0

# 2. Delete Kubernetes resources
kubectl delete -f k8s/

# 3. Destroy Terraform infrastructure
cd terraform
terraform destroy

# 4. Delete GCS bucket and contents
gsutil -m rm -r gs://$(terraform output -raw model_storage_bucket)/

# 5. Delete Docker images
gcloud container images delete ${REGISTRY}/${IMAGE}:${TAG} --force-delete-tags
```

## Troubleshooting

### Pod Startup Issues

If pods are not starting properly:

```bash
# Check pod status
kubectl get pods

# View pod details
kubectl describe pod <pod-name>

# View pod logs
kubectl logs <pod-name>
```

Common issues:
- **Insufficient resources**: Increase node size or reduce resource requests
- **Image pull errors**: Check registry permissions and image name
- **Model download failures**: Check Hugging Face token and internet connectivity

### Network Issues

If you can't access the service:

```bash
# Check service
kubectl get service llm-inference

# Test connectivity from inside the cluster
kubectl run -i --tty --rm debug --image=curlimages/curl -- curl http://llm-inference/health
```

### Model Loading Issues

If the model fails to load:

```bash
# Check logs for model-related errors
kubectl logs -l app=llm-inference

# Verify model was downloaded correctly
kubectl exec -it <pod-name> -- ls -la /models
```

### Quantization Issues

If quantization doesn't improve performance:

- Try different quantization methods (`--method gptq` or `--method optimum`)
- Try 8-bit instead of 4-bit quantization (`--bits 8`)
- Check if the model supports quantization (not all models quantize well)

### Blue-Green Deployment Issues

If the blue-green deployment isn't working properly:

```bash
# Check service selectors
kubectl get svc llm-inference -o yaml | grep -A5 selector

# Verify pod labels
kubectl get pods --show-labels
```

---

Congratulations! You've now set up a complete MLOps pipeline for LLM deployment on GKE. This solution provides infrastructure as code, containerization, streaming inference, blue-green deployments, and performance optimization through quantization.

Feel free to customize the components to fit your specific requirements and explore additional optimizations for your LLM workloads.
