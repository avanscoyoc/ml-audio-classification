#!/bin/bash

# ML Audio Classification - Kubernetes Deployment Script
# This script automates the deployment process to a Kubernetes cluster

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEFAULT_REGISTRY="gcr.io"
DEFAULT_NAMESPACE="ml-audio-classification"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
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

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check docker
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check if kubectl can connect to cluster
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Parse command line arguments
parse_args() {
    REGISTRY="$DEFAULT_REGISTRY"
    PROJECT_ID=""
    IMAGE_TAG="latest"
    NAMESPACE="$DEFAULT_NAMESPACE"
    SKIP_BUILD=false
    SKIP_PUSH=false
    DRY_RUN=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --registry)
                REGISTRY="$2"
                shift 2
                ;;
            --project-id)
                PROJECT_ID="$2"
                shift 2
                ;;
            --tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            --namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --skip-push)
                SKIP_PUSH=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown argument: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Validate required arguments
    if [[ -z "$PROJECT_ID" ]]; then
        log_error "Project ID is required. Use --project-id <project-id>"
        exit 1
    fi
    
    # Set image name
    IMAGE_NAME="$REGISTRY/$PROJECT_ID/ml-audio-classification:$IMAGE_TAG"
}

show_help() {
    cat << EOF
ML Audio Classification - Kubernetes Deployment Script

Usage: $0 [OPTIONS]

Options:
    --registry <registry>      Container registry (default: gcr.io)
    --project-id <project-id>  GCP project ID (required)
    --tag <tag>               Image tag (default: latest)
    --namespace <namespace>    Kubernetes namespace (default: ml-audio-classification)
    --skip-build              Skip Docker image build
    --skip-push               Skip Docker image push
    --dry-run                 Show what would be done without executing
    --help                    Show this help message

Examples:
    # Deploy to GCP with default settings
    $0 --project-id my-project

    # Deploy with custom tag
    $0 --project-id my-project --tag v1.2.3

    # Deploy to different registry
    $0 --registry docker.io/myorg --project-id my-project

    # Dry run to see what would happen
    $0 --project-id my-project --dry-run
EOF
}

# Build Docker image
build_image() {
    if [[ "$SKIP_BUILD" == "true" ]]; then
        log_info "Skipping Docker image build"
        return
    fi
    
    log_info "Building Docker image: $IMAGE_NAME"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would run: docker build -t $IMAGE_NAME $PROJECT_ROOT"
        return
    fi
    
    cd "$PROJECT_ROOT"
    docker build -t "$IMAGE_NAME" .
    
    log_success "Docker image built successfully"
}

# Push Docker image
push_image() {
    if [[ "$SKIP_PUSH" == "true" ]]; then
        log_info "Skipping Docker image push"
        return
    fi
    
    log_info "Pushing Docker image: $IMAGE_NAME"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would run: docker push $IMAGE_NAME"
        return
    fi
    
    # Configure docker for registry if needed
    if [[ "$REGISTRY" == "gcr.io" ]]; then
        gcloud auth configure-docker gcr.io --quiet || log_warning "Failed to configure Docker for GCR"
    fi
    
    docker push "$IMAGE_NAME"
    
    log_success "Docker image pushed successfully"
}

# Update Kubernetes manifests
update_manifests() {
    log_info "Updating Kubernetes manifests..."
    
    local temp_dir
    temp_dir=$(mktemp -d)
    
    # Copy k8s directory to temp location
    cp -r "$PROJECT_ROOT/k8s" "$temp_dir/"
    
    # Update image references in manifests
    find "$temp_dir/k8s" -name "*.yaml" -exec sed -i.bak "s|image: ml-audio-classification:latest|image: $IMAGE_NAME|g" {} \;
    
    # Update namespace if different from default
    if [[ "$NAMESPACE" != "$DEFAULT_NAMESPACE" ]]; then
        find "$temp_dir/k8s" -name "*.yaml" -exec sed -i.bak "s|namespace: $DEFAULT_NAMESPACE|namespace: $NAMESPACE|g" {} \;
    fi
    
    # Clean up backup files
    find "$temp_dir/k8s" -name "*.bak" -delete
    
    echo "$temp_dir/k8s"
}

# Deploy to Kubernetes
deploy_k8s() {
    log_info "Deploying to Kubernetes..."
    
    local manifest_dir
    manifest_dir=$(update_manifests)
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would apply manifests:"
        find "$manifest_dir" -name "*.yaml" | sort | while read -r file; do
            echo "  - $file"
        done
        rm -rf "$manifest_dir"
        return
    fi
    
    # Apply manifests in order
    kubectl apply -f "$manifest_dir/01-namespace-config.yaml"
    
    # Wait for namespace to be ready
    kubectl wait --for=condition=Active --timeout=60s namespace/"$NAMESPACE" || true
    
    # Apply remaining manifests
    kubectl apply -f "$manifest_dir/02-deployment.yaml"
    kubectl apply -f "$manifest_dir/04-service.yaml"
    
    # Clean up temp directory
    rm -rf "$manifest_dir"
    
    log_success "Kubernetes deployment completed"
}

# Check deployment status
check_deployment() {
    log_info "Checking deployment status..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would check deployment status"
        return
    fi
    
    # Wait for deployment to be ready
    kubectl wait --for=condition=available --timeout=300s deployment/ml-audio-classification -n "$NAMESPACE"
    
    # Show pod status
    kubectl get pods -n "$NAMESPACE" -l app=ml-audio-classification
    
    # Show service information
    kubectl get services -n "$NAMESPACE"
    
    log_success "Deployment is ready"
}

# Run deployment job
run_deployment_job() {
    log_info "Running deployment verification job..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would run deployment verification job"
        return
    fi
    
    # Create a simple verification job
    cat << EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ml-audio-classification-verify
  namespace: $NAMESPACE
spec:
  template:
    spec:
      containers:
      - name: verify
        image: $IMAGE_NAME
        command: ["python", "-c", "import ml_audio_classification; print('✅ Import successful')"]
      restartPolicy: Never
  backoffLimit: 3
EOF

    # Wait for job completion
    kubectl wait --for=condition=complete --timeout=120s job/ml-audio-classification-verify -n "$NAMESPACE"
    
    # Show job logs
    kubectl logs -n "$NAMESPACE" job/ml-audio-classification-verify
    
    # Clean up job
    kubectl delete job ml-audio-classification-verify -n "$NAMESPACE"
    
    log_success "Deployment verification completed"
}

# Show post-deployment information
show_post_deployment_info() {
    log_info "Post-deployment information:"
    
    cat << EOF

📋 Deployment Summary:
   Image: $IMAGE_NAME
   Namespace: $NAMESPACE
   Registry: $REGISTRY

🚀 Next Steps:

1. Check pod status:
   kubectl get pods -n $NAMESPACE

2. View logs:
   kubectl logs -n $NAMESPACE deployment/ml-audio-classification

3. Run an experiment job:
   kubectl apply -f k8s/03-jobs.yaml

4. Port forward for local access:
   kubectl port-forward -n $NAMESPACE service/ml-audio-classification-service 8080:8080

5. Scale deployment:
   kubectl scale -n $NAMESPACE deployment/ml-audio-classification --replicas=3

📖 Documentation:
   See README.md for detailed usage instructions

EOF
}

# Main execution
main() {
    log_info "Starting ML Audio Classification deployment..."
    
    check_prerequisites
    parse_args "$@"
    
    log_info "Deployment configuration:"
    log_info "  Registry: $REGISTRY"
    log_info "  Project ID: $PROJECT_ID"
    log_info "  Image: $IMAGE_NAME"
    log_info "  Namespace: $NAMESPACE"
    log_info "  Dry Run: $DRY_RUN"
    
    build_image
    push_image
    deploy_k8s
    
    if [[ "$DRY_RUN" != "true" ]]; then
        check_deployment
        run_deployment_job
        show_post_deployment_info
    fi
    
    log_success "Deployment process completed successfully!"
}

# Execute main function with all arguments
main "$@"