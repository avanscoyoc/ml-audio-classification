#!/bin/bash
set -e

# Setup Google Cloud credentials if provided
if [ ! -z "$GOOGLE_APPLICATION_CREDENTIALS" ] && [ -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "✅ Google Cloud credentials found at $GOOGLE_APPLICATION_CREDENTIALS"
else
    echo "⚠️  No Google Cloud credentials found. Set GOOGLE_APPLICATION_CREDENTIALS environment variable."
fi

# Setup logging
mkdir -p /app/logs
echo "🚀 Starting ML Audio Classification..."

# Execute the command
exec "$@"