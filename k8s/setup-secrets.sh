#!/bin/bash

# ML Audio Classification - GCS Secrets Setup Script
# This script helps you set up proper GCS credentials for Kubernetes deployment

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
if [[ ! -f "k8s/01-namespace-config.yaml" ]]; then
    log_error "Please run this script from the project root directory"
    exit 1
fi

# Get user inputs
echo "=== GCS Configuration Setup ==="
echo
read -p "Enter your GCP Project ID: " GCP_PROJECT_ID
read -p "Enter your GCS Bucket Name: " GCS_BUCKET_NAME
read -p "Enter path to your GCP Service Account JSON key file: " GCP_KEY_PATH

# Validate inputs
if [[ -z "$GCP_PROJECT_ID" ]]; then
    log_error "GCP Project ID is required"
    exit 1
fi

if [[ -z "$GCS_BUCKET_NAME" ]]; then
    log_error "GCS Bucket Name is required"
    exit 1
fi

if [[ ! -f "$GCP_KEY_PATH" ]]; then
    log_error "GCP Service Account key file not found: $GCP_KEY_PATH"
    exit 1
fi

# Base64 encode values
log_info "Encoding values for Kubernetes secrets..."
GCP_PROJECT_ID_B64=$(echo -n "$GCP_PROJECT_ID" | base64)
GCS_BUCKET_NAME_B64=$(echo -n "$GCS_BUCKET_NAME" | base64)
GCP_SERVICE_ACCOUNT_KEY_B64=$(base64 < "$GCP_KEY_PATH" | tr -d '\n')

# Create updated namespace config
log_info "Creating updated namespace configuration..."
cat > k8s/01-namespace-config-updated.yaml << EOF
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
  # GCS Configuration
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
  # Your actual encoded GCS credentials
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
  storageClassName: standard  # Adjust based on your cluster's storage classes
  resources:
    requests:
      storage: 100Gi  # Adjust based on your data needs
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

log_success "Updated configuration created: k8s/01-namespace-config-updated.yaml"
log_warning "This file contains sensitive information. Do not commit it to version control!"
log_info ""
log_info "Next steps:"
log_info "1. Review the generated file"
log_info "2. Deploy using: kubectl apply -f k8s/01-namespace-config-updated.yaml"
log_info "3. Verify secrets: kubectl get secrets -n ml-audio-classification"