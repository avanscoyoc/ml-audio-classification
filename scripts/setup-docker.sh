#!/bin/bash

# Docker Setup Script for ML Audio Classification
# This script sets up everything needed to run the application in Docker

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

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

# Check Docker installation
check_docker() {
    log_info "Checking Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker Desktop from: https://www.docker.com/products/docker-desktop"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker is not running. Please start Docker Desktop."
        exit 1
    fi
    
    log_success "Docker is installed and running"
}

# Check Docker Compose
check_docker_compose() {
    log_info "Checking Docker Compose..."
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not available. Please install Docker Compose."
        exit 1
    fi
    
    log_success "Docker Compose is available"
}

# Setup directories
setup_directories() {
    log_info "Setting up directories..."
    
    cd "$PROJECT_ROOT"
    
    local dirs=("credentials" "data" "results" "logs")
    
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        fi
    done
    
    log_success "Directories created"
}

# Setup environment file
setup_env_file() {
    log_info "Setting up environment file..."
    
    cd "$PROJECT_ROOT"
    
    if [[ ! -f ".env" ]]; then
        cp .env.example .env
        log_success "Created .env file from template"
        log_warning "Please edit .env file with your GCP project ID and other settings"
    else
        log_info ".env file already exists"
    fi
}

# Setup GCP credentials
setup_gcp_credentials() {
    log_info "Setting up GCP credentials..."
    
    cd "$PROJECT_ROOT"
    
    if [[ ! -d "credentials" ]]; then
        mkdir -p credentials
    fi
    
    if [[ ! -f "credentials/gcp-key.json" ]]; then
        log_warning "No GCP service account key found at credentials/gcp-key.json"
        echo
        echo "To set up GCP authentication:"
        echo "1. Go to Google Cloud Console: https://console.cloud.google.com/"
        echo "2. Navigate to IAM & Admin > Service Accounts"
        echo "3. Create a new service account or use existing one"
        echo "4. Download the JSON key file"
        echo "5. Save it as: $PROJECT_ROOT/credentials/gcp-key.json"
        echo "6. Ensure the service account has access to Cloud Storage"
        echo
        read -p "Press Enter to continue (or Ctrl+C to exit and set up credentials first)..."
    else
        log_success "GCP credentials found"
    fi
}

# Build Docker image
build_docker_image() {
    log_info "Building Docker image..."
    
    cd "$PROJECT_ROOT"
    
    docker build -t ml-audio-classification:latest . || {
        log_error "Docker build failed"
        exit 1
    }
    
    log_success "Docker image built successfully"
}

# Test Docker installation
test_docker_setup() {
    log_info "Testing Docker setup..."
    
    cd "$PROJECT_ROOT"
    
    # Test basic container startup
    log_info "Testing container startup..."
    docker run --rm ml-audio-classification:latest python -c "import ml_audio_classification; print('✅ Package works in Docker')" || {
        log_error "Container test failed"
        exit 1
    }
    
    # Test CLI help
    log_info "Testing CLI interface..."
    docker run --rm ml-audio-classification:latest python -m ml_audio_classification --help | head -5
    
    log_success "Docker setup test passed"
}

# Show usage examples
show_usage_examples() {
    log_info "Docker setup completed! Here's how to use it:"
    
    cat << EOF

🐳 DOCKER USAGE EXAMPLES:

1. Basic Help:
   docker-compose run --rm ml-audio-classification python -m ml_audio_classification --help

2. Single Experiment (Demo - may fail without real GCS data):
   docker-compose run --rm ml-audio-classification python -m ml_audio_classification run-experiment \\
     --models random_forest \\
     --species coyote \\
     --training-sizes 100

3. Grid Search (Demo - may fail without real GCS data):
   docker-compose run --rm ml-audio-classification python -m ml_audio_classification grid-search \\
     --models random_forest svm \\
     --species coyote bullfrog \\
     --training-sizes 50 100

4. Interactive Container (for debugging):
   docker-compose run --rm ml-audio-classification bash

5. View Results:
   docker-compose run --rm ml-audio-classification ls -la /app/results

📁 DIRECTORY STRUCTURE:
   credentials/    - GCP service account key
   data/          - Input data (mounted to container)
   results/       - Experiment results (mounted from container)
   logs/          - Application logs (mounted from container)

⚙️  CONFIGURATION:
   Edit .env file to set:
   - GCP_PROJECT_ID: Your Google Cloud project ID
   - GCS_BUCKET_NAME: Should be 'dse-staff' per CLAUDE.md
   - Other settings as needed

🔑 GCP AUTHENTICATION:
   Place your GCP service account JSON key at:
   credentials/gcp-key.json

📊 EXPECTED BEHAVIOR:
   - Without real GCS data: Commands will show interface but fail during data loading
   - With real GCS data: Full experiments will run and generate results
   - Results are saved to the mounted results/ directory

🚀 PRODUCTION READY:
   The Docker container includes all dependencies and provides a
   completely reproducible environment for running experiments.

EOF
}

# Parse command line arguments
parse_args() {
    SKIP_BUILD=false
    SKIP_TEST=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --skip-test)
                SKIP_TEST=true
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
}

show_help() {
    cat << EOF
Docker Setup Script for ML Audio Classification

Usage: $0 [OPTIONS]

Options:
    --skip-build    Skip Docker image build
    --skip-test     Skip testing the Docker setup
    --help          Show this help message

This script will:
1. Check Docker and Docker Compose installation
2. Set up required directories
3. Create environment file from template
4. Guide you through GCP credentials setup
5. Build the Docker image
6. Test the Docker setup
7. Show usage examples

EOF
}

# Main execution
main() {
    log_info "Starting Docker setup for ML Audio Classification..."
    
    parse_args "$@"
    
    check_docker
    check_docker_compose
    setup_directories
    setup_env_file
    setup_gcp_credentials
    
    if [[ "$SKIP_BUILD" != "true" ]]; then
        build_docker_image
    fi
    
    if [[ "$SKIP_TEST" != "true" ]]; then
        test_docker_setup
    fi
    
    show_usage_examples
    
    log_success "Docker setup completed successfully!"
}

# Execute main function with all arguments
main "$@"