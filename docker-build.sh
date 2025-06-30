#!/bin/bash

set -e  # Exit on error

source .env
# === Configuration ===
DOCKER_USER=$DOCKER_USER
IMAGE_NAME=visionsearch

# Pulling latest version
echo "🚀 Pulling Lastest Git Version"
git pull

# === Step 1: Get the latest Git tag ===
TAG=$(git describe --tags $(git rev-list --tags --max-count=1))

if [ -z "$TAG" ]; then
  echo "❌ No tags found. Please create a tag first."
  exit 1
fi

echo "📦 Latest tag: $TAG"

echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin

# === Step 2: Build Docker image with version tag and 'latest' ===
echo "🐳 Building Docker image: $DOCKER_USER/$IMAGE_NAME:$TAG"
docker build -t $DOCKER_USER/$IMAGE_NAME:$TAG -t $DOCKER_USER/$IMAGE_NAME:latest .

# === Step 3: Push both tags to Docker Hub ===
echo "🚀 Pushing Docker image: $DOCKER_USER/$IMAGE_NAME:$TAG"
docker push $DOCKER_USER/$IMAGE_NAME:$TAG

echo "🚀 Pushing Docker image: $DOCKER_USER/$IMAGE_NAME:latest"
docker push $DOCKER_USER/$IMAGE_NAME:latest

echo "✅ Done: $DOCKER_USER/$IMAGE_NAME tagged as $TAG and latest"