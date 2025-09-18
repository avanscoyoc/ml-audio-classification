#!/bin/bash

# ML Audio Classification - Quick Experiment Launcher
# This script helps you easily configure and launch experiments

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "🎯 ML Audio Classification - Experiment Launcher"
echo "================================================="
echo

# Default values
DEFAULT_MODELS="birdnet mobilenet"
DEFAULT_SPECIES="coyote bullfrog"
DEFAULT_SIZES="25 50 75 100"
DEFAULT_FOLDS="5"
DEFAULT_SEEDS="42 123 456"

# Get user inputs with defaults
echo "📝 Configure your experiment (press Enter for defaults):"
echo
echo "Available models: birdnet, mobilenet, vgg, perch, resnet"
read -p "Models [$DEFAULT_MODELS]: " MODELS
MODELS=${MODELS:-$DEFAULT_MODELS}

echo "Available species: coyote, bullfrog, human_vocal"
read -p "Species [$DEFAULT_SPECIES]: " SPECIES  
SPECIES=${SPECIES:-$DEFAULT_SPECIES}

echo "Training sizes (space-separated): 25 50 75 100 125 150 175 200"
read -p "Training sizes [$DEFAULT_SIZES]: " SIZES
SIZES=${SIZES:-$DEFAULT_SIZES}

read -p "CV folds [$DEFAULT_FOLDS]: " FOLDS
FOLDS=${FOLDS:-$DEFAULT_FOLDS}

echo "Random seeds for confidence intervals (space-separated)"
read -p "Random seeds [$DEFAULT_SEEDS]: " SEEDS
SEEDS=${SEEDS:-$DEFAULT_SEEDS}

echo
echo -e "${BLUE}📋 Experiment Configuration:${NC}"
echo "Models: $MODELS"
echo "Species: $SPECIES"  
echo "Training sizes: $SIZES"
echo "CV folds: $FOLDS"
echo "Random seeds: $SEEDS"
echo

# Confirm
read -p "Deploy this experiment? (y/N): " CONFIRM
if [[ ! $CONFIRM =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

# Generate unique job name with timestamp
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
JOB_NAME="ml-audio-experiment-$TIMESTAMP"

echo -e "${BLUE}🚀 Deploying experiment: $JOB_NAME${NC}"

# Function to convert space-separated values to YAML array format
generate_yaml_args() {
    local flag="$1"
    local values="$2"
    
    echo "        - \"$flag\""
    for value in $values; do
        echo "        - \"$value\""
    done
}

# Generate the complete args section
ARGS_YAML=""
ARGS_YAML+="$(generate_yaml_args "--models" "$MODELS")"$'\n'
ARGS_YAML+="$(generate_yaml_args "--species" "$SPECIES")"$'\n'
ARGS_YAML+="$(generate_yaml_args "--training-sizes" "$SIZES")"$'\n'
ARGS_YAML+="        - \"--cv-folds\""$'\n'
ARGS_YAML+="        - \"$FOLDS\""$'\n'
ARGS_YAML+="$(generate_yaml_args "--seeds" "$SEEDS")"$'\n'
ARGS_YAML+="        - \"--output-dir\""$'\n'
ARGS_YAML+="        - \"/app/results\""

# Create and deploy the job
cat << EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: $JOB_NAME
  namespace: ml-audio-classification
  labels:
    app: ml-audio-classification
    job-type: experiment
spec:
  template:
    metadata:
      labels:
        app: ml-audio-classification
        job-type: experiment
    spec:
      containers:
      - name: ml-audio-classification
        image: gcr.io/dse-staff/ml-audio-classification:v1.0
        imagePullPolicy: IfNotPresent
        
        envFrom:
        - configMapRef:
            name: ml-audio-classification-config
        - secretRef:
            name: ml-audio-classification-secrets
        
        env:
        - name: GOOGLE_APPLICATION_CREDENTIALS
          value: "/app/credentials/gcp-key.json"
        - name: PYTHONPATH
          value: "/app/src"
        - name: PYTHONUNBUFFERED
          value: "1"
        - name: TESTING_MODE
          value: "false"
        
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: results-volume
          mountPath: /app/results
        - name: gcp-credentials
          mountPath: /app/credentials
          readOnly: true
        
        command: ["python", "-m", "ml_audio_classification", "run-experiment"]
        args: 
$ARGS_YAML
      
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: ml-audio-classification-data
      - name: results-volume
        persistentVolumeClaim:
          claimName: ml-audio-classification-results
      - name: gcp-credentials
        secret:
          secretName: ml-audio-classification-secrets
          items:
          - key: GCP_SERVICE_ACCOUNT_KEY
            path: gcp-key.json
      
      imagePullSecrets:
      - name: gcr-json-key
      
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
      
      restartPolicy: Never
  
  backoffLimit: 2
  ttlSecondsAfterFinished: 86400
EOF

echo -e "${GREEN}✅ Experiment deployed successfully!${NC}"
echo
echo "📊 Monitor your experiment:"
echo "  kubectl logs -n ml-audio-classification job/$JOB_NAME -f"
echo
echo "📋 Check status:"
echo "  kubectl get jobs -n ml-audio-classification"
echo "  kubectl describe job -n ml-audio-classification $JOB_NAME"
echo
echo "📁 View results (when complete):"
echo "  kubectl get pods -n ml-audio-classification"
echo "  kubectl exec -n ml-audio-classification [pod-name] -- ls -la /app/results"