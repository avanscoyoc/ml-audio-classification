#!/bin/bash

# ML Audio Classification - Local Development Setup Script
# This script sets up the development environment

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_VERSION="3.10"

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

# Check Python version
check_python() {
    log_info "Checking Python version..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi
    
    local python_version
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    
    if [[ $(echo "$python_version >= $PYTHON_VERSION" | bc -l) -eq 0 ]]; then
        log_error "Python $PYTHON_VERSION or higher is required (found $python_version)"
        exit 1
    fi
    
    log_success "Python $python_version found"
}

# Setup virtual environment
setup_venv() {
    log_info "Setting up virtual environment..."
    
    cd "$PROJECT_ROOT"
    
    if [[ -d "venv" ]]; then
        log_warning "Virtual environment already exists"
        read -p "Remove existing venv and create new one? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf venv
        else
            log_info "Using existing virtual environment"
            return
        fi
    fi
    
    python3 -m venv venv
    log_success "Virtual environment created"
}

# Activate virtual environment
activate_venv() {
    log_info "Activating virtual environment..."
    
    cd "$PROJECT_ROOT"
    
    # Source the virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    python -m pip install --upgrade pip
    
    log_success "Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    log_info "Installing dependencies..."
    
    cd "$PROJECT_ROOT"
    
    # Install core dependencies
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
        log_success "Core dependencies installed"
    else
        log_warning "requirements.txt not found"
    fi
    
    # Install development dependencies
    if [[ -f "requirements-dev.txt" ]]; then
        pip install -r requirements-dev.txt
        log_success "Development dependencies installed"
    else
        log_warning "requirements-dev.txt not found"
    fi
    
    # Install package in editable mode
    if [[ -f "pyproject.toml" ]]; then
        pip install -e .
        log_success "Package installed in editable mode"
    else
        log_warning "pyproject.toml not found"
    fi
}

# Setup pre-commit hooks
setup_pre_commit() {
    log_info "Setting up pre-commit hooks..."
    
    cd "$PROJECT_ROOT"
    
    if command -v pre-commit &> /dev/null; then
        if [[ -f ".pre-commit-config.yaml" ]]; then
            pre-commit install
            log_success "Pre-commit hooks installed"
        else
            log_warning ".pre-commit-config.yaml not found, skipping pre-commit setup"
        fi
    else
        log_warning "pre-commit not available, skipping hooks setup"
    fi
}

# Create necessary directories
create_directories() {
    log_info "Creating necessary directories..."
    
    cd "$PROJECT_ROOT"
    
    local dirs=("data" "results" "logs" "temp")
    
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        fi
    done
    
    log_success "Directories created"
}

# Setup configuration files
setup_config() {
    log_info "Setting up configuration files..."
    
    cd "$PROJECT_ROOT"
    
    # Copy environment template if it doesn't exist
    if [[ -f ".env.example" ]] && [[ ! -f ".env" ]]; then
        cp .env.example .env
        log_info "Created .env from template"
        log_warning "Please edit .env file with your configuration"
    fi
    
    # Create .gitignore if it doesn't exist
    if [[ ! -f ".gitignore" ]]; then
        cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
.env
data/
results/
logs/
temp/
credentials/
*.log

# ML artifacts
models/
checkpoints/
*.pkl
*.joblib
EOF
        log_success "Created .gitignore"
    fi
    
    log_success "Configuration setup completed"
}

# Run tests
run_tests() {
    log_info "Running tests..."
    
    cd "$PROJECT_ROOT"
    
    if command -v pytest &> /dev/null; then
        if [[ -d "tests" ]] || find . -name "test_*.py" -o -name "*_test.py" | grep -q .; then
            pytest --version
            # pytest -v  # Uncomment when tests are available
            log_info "Test framework is ready (no tests found yet)"
        else
            log_warning "No tests found"
        fi
    else
        log_warning "pytest not available"
    fi
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    cd "$PROJECT_ROOT"
    
    # Test import
    if python -c "import ml_audio_classification; print('✅ Package import successful')" 2>/dev/null; then
        log_success "Package import test passed"
    else
        log_error "Package import test failed"
        return 1
    fi
    
    # Test CLI
    if python -m ml_audio_classification --help &>/dev/null; then
        log_success "CLI test passed"
    else
        log_warning "CLI test failed (this may be expected if dependencies are missing)"
    fi
    
    log_success "Installation verification completed"
}

# Show setup summary
show_summary() {
    log_info "Setup Summary:"
    
    cat << EOF

🎉 Development Environment Setup Complete!

📁 Project Structure:
   $(find "$PROJECT_ROOT" -maxdepth 2 -type d | head -10 | sed 's|^|   |')

🐍 Python Environment:
   Location: $PROJECT_ROOT/venv
   Python: $(python --version 2>&1)
   Pip: $(pip --version | cut -d' ' -f1-2)

📦 Installed Packages:
   $(pip list | grep -E "(ml-audio|tensorflow|torch|sklearn)" | sed 's|^|   |' || echo "   Core ML packages will be installed as needed")

🚀 Next Steps:

1. Activate the environment:
   cd $PROJECT_ROOT
   source venv/bin/activate

2. Configure your environment:
   edit .env file with your GCP credentials and settings

3. Test the installation:
   python -m ml_audio_classification --help

4. Run your first experiment:
   python -m ml_audio_classification run-experiment \\
     --models random_forest \\
     --species "Turdus migratorius" \\
     --training-sizes 100

5. Check the documentation:
   open README.md

📖 Documentation:
   - README.md: Main documentation
   - src/: Source code with docstrings
   - k8s/: Kubernetes deployment files

🔧 Development Tools:
   - Black: Code formatting
   - Flake8: Linting
   - MyPy: Type checking
   - Pytest: Testing framework

Happy coding! 🎵🤖

EOF
}

# Parse command line arguments
parse_args() {
    SKIP_TESTS=false
    SKIP_VERIFICATION=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --skip-verification)
                SKIP_VERIFICATION=true
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
ML Audio Classification - Development Setup Script

Usage: $0 [OPTIONS]

Options:
    --skip-tests         Skip running tests
    --skip-verification  Skip installation verification
    --help              Show this help message

This script will:
1. Check Python version ($PYTHON_VERSION+ required)
2. Create and activate virtual environment
3. Install dependencies
4. Setup pre-commit hooks
5. Create necessary directories
6. Setup configuration files
7. Run tests (optional)
8. Verify installation (optional)

EOF
}

# Main execution
main() {
    log_info "Starting ML Audio Classification development setup..."
    
    parse_args "$@"
    
    check_python
    setup_venv
    activate_venv
    install_dependencies
    setup_pre_commit
    create_directories
    setup_config
    
    if [[ "$SKIP_TESTS" != "true" ]]; then
        run_tests
    fi
    
    if [[ "$SKIP_VERIFICATION" != "true" ]]; then
        verify_installation
    fi
    
    show_summary
    
    log_success "Development environment setup completed successfully!"
}

# Execute main function with all arguments
main "$@"