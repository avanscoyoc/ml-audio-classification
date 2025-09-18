#!/bin/bash

# ML Audio Classification Setup Script
# This script helps configure the application for your GCP environment

set -e

echo "🎵 ML Audio Classification Setup"
echo "==============================="
echo

# Check if .env already exists
if [ -f ".env" ]; then
    echo "⚠️  .env file already exists."
    read -p "Do you want to overwrite it? (y/N): " overwrite
    if [[ ! $overwrite =~ ^[Yy]$ ]]; then
        echo "Setup cancelled."
        exit 0
    fi
fi

# Get user input
echo "Please provide your Google Cloud configuration:"
echo

read -p "GCP Project ID: " gcp_project_id
read -p "GCS Bucket Name: " gcs_bucket_name

# Optional settings
echo
echo "Optional settings (press Enter for defaults):"
read -p "Testing Mode (true/false) [false]: " testing_mode
testing_mode=${testing_mode:-false}

read -p "Log Level (DEBUG/INFO/WARNING/ERROR) [INFO]: " log_level
log_level=${log_level:-INFO}

read -p "Max Workers [4]: " max_workers
max_workers=${max_workers:-4}

# Create .env file
cat > .env << EOF
# ML Audio Classification Configuration
# Generated on $(date)

# Google Cloud Platform Settings (REQUIRED)
GCP_PROJECT_ID=$gcp_project_id
GCS_BUCKET_NAME=$gcs_bucket_name
GOOGLE_APPLICATION_CREDENTIALS=./credentials/gcp-key.json

# Application Settings
TESTING_MODE=$testing_mode
LOG_LEVEL=$log_level
MAX_WORKERS=$max_workers

# Optional: GPU support (uncomment if using nvidia-docker)
# NVIDIA_VISIBLE_DEVICES=all
EOF

echo
echo "✅ Configuration saved to .env"
echo

# Check for credentials directory
if [ ! -d "credentials" ]; then
    echo "📁 Creating credentials directory..."
    mkdir -p credentials
fi

# Check for service account key
if [ ! -f "credentials/gcp-key.json" ]; then
    echo
    echo "🔑 Next steps:"
    echo "1. Download your GCP service account key to credentials/gcp-key.json"
    echo "2. Make sure your bucket has the required structure:"
    echo "   $gcs_bucket_name/soundhub/data/audio/{species}/{data,non_target}/"
    echo "3. Run: docker-compose build"
    echo "4. Test: docker-compose run --rm ml-audio-classification python -m ml_audio_classification --help"
else
    echo "✅ Service account key found"
    echo
    echo "🚀 Ready to run:"
    echo "   docker-compose build"
    echo "   docker-compose run --rm ml-audio-classification python -m ml_audio_classification --help"
fi

echo
echo "🔗 For more information, see README.md"