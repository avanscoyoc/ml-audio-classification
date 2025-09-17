#!/bin/bash

# Quick Start Script for ML Audio Classification
# This script demonstrates running experiments according to CLAUDE.md specifications

set -e

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
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

main() {
    cd "$PROJECT_ROOT"
    
    log_info "Starting ML Audio Classification Quick Demo"
    log_info "Using species: coyote, bullfrog, human_vocal (as per CLAUDE.md)"
    
    # Check if virtual environment is activated
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        log_warning "No virtual environment detected. Attempting to activate..."
        if [[ -f "venv/bin/activate" ]]; then
            source venv/bin/activate
            log_success "Activated virtual environment"
        else
            log_warning "No venv found. Please run ./scripts/setup-dev.sh first"
            exit 1
        fi
    fi
    
    # Create results directory
    mkdir -p "results/demo_$TIMESTAMP"
    
    log_info "Running single experiment with coyote detection..."
    
    # Single experiment - Coyote detection with 2 models
    python -m ml_audio_classification run-experiment \
        --models random_forest vgg \
        --species coyote \
        --training-sizes 50 100 \
        --cv-folds 3 \
        --output-dir "results/demo_$TIMESTAMP/single_experiment" || {
        log_warning "Single experiment failed (this is expected if GCS data is not accessible)"
    }
    
    log_info "Running mini grid search across species..."
    
    # Grid search - All 3 species with selected models
    python -m ml_audio_classification grid-search \
        --models random_forest svm \
        --species coyote bullfrog human_vocal \
        --training-sizes 50 100 \
        --cv-folds 3 \
        --max-concurrent 2 \
        --output-dir "results/demo_$TIMESTAMP/grid_search" || {
        log_warning "Grid search failed (this is expected if GCS data is not accessible)"
    }
    
    log_success "Demo completed! Check results in: results/demo_$TIMESTAMP/"
    
    echo
    log_info "To run with real data, ensure:"
    echo "  1. GCP credentials are configured"
    echo "  2. Access to dse-staff/soundhub bucket"
    echo "  3. Species data exists in: soundhub/data/audio/{species}/data/"
    echo "  4. Perch data exists in: soundhub/data/audio/{species}/data_5s/"
    
    echo
    log_info "Available species (per CLAUDE.md):"
    echo "  - coyote"
    echo "  - bullfrog" 
    echo "  - human_vocal"
    
    echo
    log_info "Available models:"
    echo "  - random_forest (fast, MFCC features)"
    echo "  - svm (fast, MFCC features)"
    echo "  - vgg (CNN, spectrograms)"
    echo "  - mobilenet (lightweight CNN, spectrograms)"
    echo "  - resnet (deep CNN, spectrograms)"
    echo "  - birdnet (specialized, pre-trained embeddings)"
    echo "  - perch (specialized, 5s audio segments)"
    
    echo
    log_info "Example production command:"
    echo "  python -m ml_audio_classification grid-search \\"
    echo "    --models random_forest svm vgg mobilenet resnet birdnet perch \\"
    echo "    --species coyote bullfrog human_vocal \\"
    echo "    --training-sizes 50 100 200 300 \\"
    echo "    --cv-folds 5 \\"
    echo "    --max-concurrent 3"
}

# Show help if requested
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    cat << EOF
ML Audio Classification Quick Start Demo

This script demonstrates the experiment functionality according to CLAUDE.md specifications.

Usage: $0

The script will:
1. Run a single experiment with coyote detection
2. Run a mini grid search across all three species
3. Show example commands for production use

Species (per CLAUDE.md):
  - coyote
  - bullfrog
  - human_vocal

Models:
  - random_forest, svm (traditional ML with MFCC)
  - vgg, mobilenet, resnet (deep learning with spectrograms)
  - birdnet, perch (specialized models)

Note: Experiments may fail if GCS data is not accessible, but will demonstrate the interface.

EOF
    exit 0
fi

# Run main function
main "$@"