#!/bin/bash

# ML Audio Classification - GKE Cluster Setup
# Project: dse-staff

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

echo "=========================================="
echo "  GKE Cluster Setup for ML Audio Classification"
echo "  Project: dse-staff"
echo "=========================================="
echo

# Configuration
PROJECT_ID="dse-staff"
CLUSTER_NAME="ml-audio-cluster"
ZONE="us-central1-a"  # Change if you prefer a different zone
REGION="us-central1"
NODE_COUNT=3
MACHINE_TYPE="e2-standard-4"  # 4 vCPUs, 16GB RAM - good for ML workloads

log_info "🔧 Configuration:"
echo "Project ID: $PROJECT_ID"
echo "Cluster Name: $CLUSTER_NAME"
echo "Zone: $ZONE"
echo "Machine Type: $MACHINE_TYPE"
echo "Node Count: $NODE_COUNT"
echo

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    log_error "gcloud CLI is not installed. Please install it first:"
    log_info "https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if user is authenticated
log_info "🔐 Checking gcloud authentication..."
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
    log_warning "You need to authenticate with gcloud first."
    read -p "Run 'gcloud auth login' now? (y/n): " auth_choice
    if [[ $auth_choice =~ ^[Yy]$ ]]; then
        gcloud auth login
    else
        log_error "Authentication required. Please run: gcloud auth login"
        exit 1
    fi
fi

ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
log_success "Authenticated as: $ACTIVE_ACCOUNT"

# Set the project
log_info "🎯 Setting project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Enable required APIs
log_info "🔌 Enabling required Google Cloud APIs..."
gcloud services enable container.googleapis.com
gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com

# Check if cluster already exists
log_info "🔍 Checking if cluster already exists..."
if gcloud container clusters describe $CLUSTER_NAME --zone=$ZONE &>/dev/null; then
    log_warning "Cluster '$CLUSTER_NAME' already exists in zone '$ZONE'"
    read -p "Do you want to use the existing cluster? (y/n): " use_existing
    if [[ $use_existing =~ ^[Yy]$ ]]; then
        log_info "Using existing cluster..."
    else
        log_error "Please choose a different cluster name or delete the existing cluster."
        exit 1
    fi
else
    # Create the GKE cluster
    log_info "🚀 Creating GKE cluster '$CLUSTER_NAME'..."
    log_warning "This will take 5-10 minutes..."
    
    gcloud container clusters create $CLUSTER_NAME \
        --zone=$ZONE \
        --machine-type=$MACHINE_TYPE \
        --num-nodes=$NODE_COUNT \
        --network=ml-audio-network \
        --subnetwork=ml-audio-subnet \
        --enable-autoscaling \
        --min-nodes=1 \
        --max-nodes=5 \
        --enable-autorepair \
        --enable-autoupgrade \
        --disk-size=50GB \
        --disk-type=pd-standard \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --no-enable-basic-auth \
        --no-issue-client-certificate \
        --enable-network-policy \
        --addons=HorizontalPodAutoscaling,HttpLoadBalancing

    log_success "✅ GKE cluster created successfully!"
fi

# Get cluster credentials for kubectl
log_info "🔑 Configuring kubectl to connect to the cluster..."
gcloud container clusters get-credentials $CLUSTER_NAME --zone=$ZONE

# Verify kubectl connection
log_info "🔍 Verifying kubectl connection..."
kubectl cluster-info
kubectl get nodes

log_success "✅ kubectl is now connected to your GKE cluster!"

# Create a container registry for your Docker images
log_info "📦 Setting up Container Registry..."
gcloud services enable containerregistry.googleapis.com

# Configure Docker to use gcloud as a credential helper
log_info "🐳 Configuring Docker for GCR..."
gcloud auth configure-docker

log_success "✅ GKE cluster setup complete!"
echo
echo "🎯 Next Steps:"
echo "1. Push your Docker image to GCR:"
echo "   docker tag ml-audio-classification:k8s gcr.io/$PROJECT_ID/ml-audio-classification:latest"
echo "   docker push gcr.io/$PROJECT_ID/ml-audio-classification:latest"
echo
echo "2. Deploy your application:"
echo "   ./k8s/deploy-complete.sh"
echo
echo "3. Launch your experiment:"
echo "   ./k8s/launch-experiment.sh"
echo
echo "📋 Cluster Information:"
echo "Project: $PROJECT_ID"
echo "Cluster: $CLUSTER_NAME"
echo "Zone: $ZONE"
echo "Nodes: $NODE_COUNT x $MACHINE_TYPE"
echo
echo "🔧 Useful commands:"
echo "View cluster: gcloud container clusters describe $CLUSTER_NAME --zone=$ZONE"
echo "Delete cluster: gcloud container clusters delete $CLUSTER_NAME --zone=$ZONE"
echo "View costs: https://console.cloud.google.com/billing"