#!/bin/bash

# Quick test with digest-based image
kubectl apply -f - << EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: ml-audio-experiment-test
  namespace: ml-audio-classification
spec:
  template:
    spec:
      containers:
      - name: ml-audio-classification
        image: gcr.io/dse-staff/ml-audio-classification@sha256:6531ec39800abd5dde840ae8c0ea1d52a79b5913a35fe0df8136fb335e767
        imagePullPolicy: Always
        command: ["python", "-c", "print('Hello from container!'); import ml_audio_classification; print('Package loaded successfully')"]
        env:
        - name: PYTHONPATH
          value: "/app/src"
      restartPolicy: Never
  backoffLimit: 1
EOF