#!/bin/bash

# ML Audio Classification - Complete Kubernetes Deployment
# Project: dse-staff
# This script deploys your ML application to Kubernetes with real GCS integration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [[ ! -f "credentials/gcp-key.json" ]]; then
    log_error "GCP key file not found at credentials/gcp-key.json"
    exit 1
fi

echo "===========================================" 
echo "  ML Audio Classification - K8s Deployment"
echo "  Project: dse-staff"
echo "==========================================="
echo

# Get bucket name from user
echo "📋 Configuration Setup"
read -p "Enter your GCS Bucket Name (where your audio data is stored): " GCS_BUCKET_NAME
if [[ -z "$GCS_BUCKET_NAME" ]]; then
    log_error "GCS Bucket Name is required"
    exit 1
fi

# Fixed project ID from your credentials
GCP_PROJECT_ID="dse-staff"
log_success "Using GCP Project: $GCP_PROJECT_ID"
log_success "Using GCS Bucket: $GCS_BUCKET_NAME"

# Check kubectl connection
log_info "Checking Kubernetes cluster connection..."
if ! kubectl cluster-info &> /dev/null; then
    log_error "Cannot connect to Kubernetes cluster. Please check your kubectl configuration."
    log_info "Run: kubectl config get-contexts"
    exit 1
fi

CURRENT_CONTEXT=$(kubectl config current-context)
log_success "Connected to cluster: $CURRENT_CONTEXT"

# Encode values for Kubernetes secrets
log_info "🔐 Encoding credentials for Kubernetes..."
GCP_PROJECT_ID_B64=$(echo -n "$GCP_PROJECT_ID" | base64)
GCS_BUCKET_NAME_B64=$(echo -n "$GCS_BUCKET_NAME" | base64)
GCP_SERVICE_ACCOUNT_KEY_B64=$(base64 < credentials/gcp-key.json | tr -d '\n')

# Create the namespace and secrets
log_info "🚀 Creating Kubernetes resources..."
cat << EOF | kubectl apply -f -
apiVersion: v1
kind: Namespace
metadata:
  name: ml-audio-classification
  labels:
    name: ml-audio-classification
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-audio-classification-config
  namespace: ml-audio-classification
data:
  LOG_LEVEL: "INFO"
  MAX_WORKERS: "4"
  AUDIO_SAMPLE_RATE: "22050"
  RANDOM_SEED: "42"
  DEFAULT_CV_FOLDS: "5"
  GCP_PROJECT_ID: "$GCP_PROJECT_ID"
  GCS_BUCKET_NAME: "$GCS_BUCKET_NAME"
  GCS_DATA_PREFIX: "audio-data/"
  GCS_RESULTS_PREFIX: "results/"
---
apiVersion: v1
kind: Secret
metadata:
  name: ml-audio-classification-secrets
  namespace: ml-audio-classification
type: Opaque
data:
  GCP_PROJECT_ID: $GCP_PROJECT_ID_B64
  GCS_BUCKET_NAME: $GCS_BUCKET_NAME_B64
  GCP_SERVICE_ACCOUNT_KEY: $GCP_SERVICE_ACCOUNT_KEY_B64
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ml-audio-classification-data
  namespace: ml-audio-classification
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: standard
  resources:
    requests:
      storage: 100Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ml-audio-classification-results
  namespace: ml-audio-classification
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: standard
  resources:
    requests:
      storage: 50Gi
EOF

log_success "✅ Namespace, ConfigMap, Secrets, and PVCs created!"

# Verify the secrets
log_info "🔍 Verifying deployment..."
kubectl get namespace ml-audio-classification
kubectl get secrets -n ml-audio-classification
kubectl get configmap -n ml-audio-classification
kubectl get pvc -n ml-audio-classification

log_success "✅ Kubernetes configuration complete!"
echo
echo "🎯 Next Steps:"
echo "1. Run BirdNET experiment:     kubectl apply -f k8s/experiment-jobs.yaml"
echo "2. Check job status:           kubectl get jobs -n ml-audio-classification"
echo "3. View logs:                  kubectl logs -n ml-audio-classification job/ml-audio-classification-experiment-birdnet-only"
echo "4. Check results:              kubectl exec -n ml-audio-classification [pod-name] -- ls -la /app/results"
echo
echo "📊 Available Experiments:"
echo "- BirdNET only (recommended):  ml-audio-classification-experiment-birdnet-only"
echo "- BirdNET + MobileNet:         ml-audio-classification-experiment-birdnet-mobilenet"
echo
echo "🔧 Troubleshooting commands:"
echo "- kubectl describe job -n ml-audio-classification [job-name]"
echo "- kubectl get pods -n ml-audio-classification"
echo "- kubectl describe pod -n ml-audio-classification [pod-name]"