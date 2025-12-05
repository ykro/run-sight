#!/bin/bash

# Configuration
PROJECT_ID=$(gcloud config get-value project)
APP_NAME="runsight-web"
REGION="us-central1"
TAG="gcr.io/$PROJECT_ID/$APP_NAME"

echo "Deploying [$APP_NAME] to Cloud Run..."

# 1. Build Container
echo "🔨 Building Container..."
gcloud builds submit --tag $TAG .

# 2. Deploy to Cloud Run
echo "☁️ Deploying Service..."
# Note: We are allowing unauthenticated invocations for a public demo. 
# For private usage, remove '--allow-unauthenticated'.
# We pass environment variables from the local .env if possible, 
# BUT for security, you should set them in Secret Manager.
# Here we'll ask the user to ensure GEMINI_API_KEY is set via the UI or console vars
# Or we can read it from local .env to inject it (Not recommended for prod CI, but fine for local script)

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

gcloud run deploy $APP_NAME \
  --image $TAG \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --cpu 2 \
  --set-env-vars GEMINI_API_KEY=$GEMINI_API_KEY

echo "✅ Deployment Complete!"
